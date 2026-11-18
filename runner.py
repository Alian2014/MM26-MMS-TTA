import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
from loguru import logger
import sys
from IPython import embed
from utils import AverageMeter
import copy
import json
from torch.utils.data import DataLoader
from collections import defaultdict

import faiss
import faiss.contrib.torch_utils
import ot
import os


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        dest='config',
        required=True,
        help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument(
        '--wandb-log',
        dest='wandb',
        action='store_true',
        help=
        'Whether you want to log to wandb. Include this flag to enable logging.'
    )
    parser.add_argument(
        '--datasets',
        dest='datasets',
        type=str,
        required=True,
        help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S"
    )
    parser.add_argument(
        '--features-root',
        dest='features_root',
        type=str,
        default='./features/vit_b16',
        help='Path to the feature directory. Default is ./features/vit_b16')
    parser.add_argument(
        '--cupl-root',
        dest='cupl_root',
        type=str,
        default='./descriptions/CuPL_prompts_{}.json',
        help=
        'Path to the descriptions file. Default is "./descriptions/CuPL_prompts_{}.json"')
    parser.add_argument('--backbone',
                        dest='backbone',
                        type=str,
                        choices=['RN50', 'ViT-B/16'],
                        required=True,
                        help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--views', type=int, default=64)
    parser.add_argument('--tau_gate', type=float, default=100)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1)
    args = parser.parse_args()

    return args


def update_cache(cache,
                 pred,
                 features_loss,
                 shot_capacity,
                 include_prob_map=False,
                 fifo=True):
    with torch.no_grad():
        # item = [feature, loss] (+ optional prob_map)
        # prob_map.shape == [1, C], feature.shape == [1, D]
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            cache[pred].append(item)
            if fifo:
                # FIFO eviction
                if len(cache[pred]) > shot_capacity:
                    cache[pred] = cache[pred][1:]
            # Keep lowest-loss samples
            else:
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
                cache[pred] = cache[pred][:shot_capacity]
        else:
            cache[pred] = [item]


def get_pre_tempalte(dataset):
    # Load external vector DB
    if 'food101' in dataset:
        pre_template = 'a photo of a {}, a type of food. '
    elif 'fgvc' in dataset:
        pre_template = 'a photo of a {} from Fine-Grained Visual Classification of Aircraft. '
    elif 'stanford_cars' in dataset:
        pre_template = 'a photo of a {}. '
    elif 'caltech101' in dataset:
        pre_template = 'a photo of a {}. '
    elif 'caltech256' in dataset:
        pre_template = 'a photo of a {}. '
    elif 'ucf101' in dataset:
        pre_template = 'a photo of a person doing {}. '
    elif 'sun397' in dataset:
        pre_template = 'a photo of a {}. '
    elif 'resisc45' in dataset:
        pre_template = 'a satellite photo of the {}. '
    elif 'cub200' in dataset:
        pre_template = 'a photo of a {}, a type of bird. '
    elif 'flowers102' in dataset:
        pre_template = 'a photo of a {}, a type of flower. '
    elif 'oxford_pets' in dataset:
        pre_template = 'a photo of a {}, a type of pet. '
    elif 'dtd' in dataset:
        pre_template = 'a photo of a {}, which is a type of texture, or a type of pattern. '
    elif 'oxford_flowers' in dataset:
        pre_template = 'a photo of a {}, a type of flower. '
    elif 'I' in dataset:
        pre_template = "a photo of the {}. "
    elif 'V' in dataset:
        pre_template = "a photo of the {}. "
    elif 'A' in dataset:
        pre_template = "a photo of the {}. "
    elif 'R' in dataset:
        pre_template = "a rendition photo of the {}. "
    elif 'S' in dataset:
        pre_template = "a sketch photo of a {}. "

    return pre_template


def _get_global_prototype(sample_cls_name2desc_id, device):
    # If there is no top-k descriptions,
    if len(sample_cls_name2desc_id.values()) == 0:
        return None
    texts_embeds = []
    with torch.no_grad():
        texts_embeds += list(torch.unbind(description_embeddings[np.concatenate(list(sample_cls_name2desc_id.values())).tolist()], dim=0))
        texts_embeds = torch.stack(texts_embeds, dim=0)
        cls_txt_embeds = texts_embeds.mean(dim=0)
        cls_txt_embeds /= cls_txt_embeds.norm()
    tst_prototype = cls_txt_embeds.unsqueeze(1).to(device)
    return tst_prototype


def _get_set_prototypes(sample_cls_name2desc_id, device):
    selected_prompts_embeds = []
    selected_prompts_keys = []
    # If there is no top-k descriptions,
    if len(sample_cls_name2desc_id.values()) == 0:
        return None, None
    with torch.no_grad():
        for key, prompt_ids in sample_cls_name2desc_id.items():
            texts_embeds = []
            texts_embeds += list(torch.unbind(description_embeddings[prompt_ids], dim=0))
            texts_embeds = torch.stack(texts_embeds, dim=0)
            cls_txt_embeds = texts_embeds.mean(dim=0)
            cls_txt_embeds /= cls_txt_embeds.norm()
            selected_prompts_embeds.append(cls_txt_embeds)
            selected_prompts_keys.append(key)
        selected_prompts_keys = torch.Tensor(selected_prompts_keys).to(device=device, dtype=torch.int64)
        return torch.stack(selected_prompts_embeds, dim=1).to(device), F.one_hot(selected_prompts_keys, num_classes=len(CLASSNAMES))


def _compute_prototypes(sample_id, sample_topk_desc_ids, device, threshold=5):
    # 1. Preprocess extracted description IDs
    sample_cls_name2desc_id = defaultdict(list)
    selected_classes = NAMES[sample_topk_desc_ids]
    for x, y in zip(selected_classes, sample_topk_desc_ids):
        sample_cls_name2desc_id[x.item()].append(y.item())
    # Drop classes with too few unique descriptions
    keys_to_delete = [key for key, val in sample_cls_name2desc_id.items() if len(set(val)) < threshold]
    for key in keys_to_delete:
        del sample_cls_name2desc_id[key]

    # 2. Compute prototypes using the preprocessed descriptions
    tst_prototype = _get_global_prototype(sample_cls_name2desc_id, device)
    set_prototypes, set_pseudo_labels = _get_set_prototypes(sample_cls_name2desc_id, device)
    return sample_id, tst_prototype, set_prototypes, set_pseudo_labels


def compute_prototypes(topk_desc_ids, device):
    batch_tst_results = defaultdict(dict)
    batch_set_results = defaultdict(dict)
    batch_set_pl_results = defaultdict(dict)
    batch_id = 0
    args_lst = [(batch_id, topk_desc_ids[batch_id], device, 5)]
    batch_id, tst_prototype, set_prototypes, set_pseudo_labels = _compute_prototypes(*args_lst[0])
    batch_tst_results[batch_id] = tst_prototype
    batch_set_results[batch_id] = set_prototypes
    batch_set_pl_results[batch_id] = set_pseudo_labels
    batch_tst_results = dict(sorted(batch_tst_results.items()))
    batch_set_results = dict(sorted(batch_set_results.items()))
    batch_set_pl_results = dict(sorted(batch_set_pl_results.items()))
    return batch_tst_results, batch_set_results, batch_set_pl_results


def get_img_desc_sim(tst_x_embeds, aug_x_embeds, description_encodings, args):
    # Test Image Embedding
    tst_img_desc_sim = [tst_x_embeds @ desc.T for _, desc in description_encodings.items()]
    tst_img_desc_sim = torch.cat(tst_img_desc_sim, dim=1)


    # Augmented Images Embeddings
    aug_img_desc_sim = [aug_x_embeds @ desc.T for _, desc in description_encodings.items()]
    aug_img_desc_sim = torch.cat(aug_img_desc_sim, dim=-1)
    aug_x_embeds = aug_x_embeds.reshape(args.batch_size, args.views - 1, -1)  # [BS, AS, ED]
    aug_img_desc_sim = aug_img_desc_sim.reshape(args.batch_size, args.views - 1, -1)  # [BS, AS, FD]

    if aug_img_desc_sim is not None and aug_img_desc_sim.numel() > 0 and aug_img_desc_sim.shape[0] > 0:
        augmented_desc_sims = torch.cat([tst_img_desc_sim.unsqueeze(1), aug_img_desc_sim], dim=1)# [BS, (AS+1), FD]
    else:
        augmented_desc_sims = tst_img_desc_sim.unsqueeze(1)

    return tst_img_desc_sim, aug_img_desc_sim, augmented_desc_sims, aug_x_embeds


def _compute_ot_based_relevance_score(id,
                                      tst_x_embed,
                                      aug_x_embeds,
                                      retrieved_embeds_lst,
                                      tst_prototype,
                                      set_prototypes,
                                      cfg,
                                      device):
    set_distance_lst = []
    if aug_x_embeds is not None and aug_x_embeds.numel() > 0:
        augmented_tst_embeds = torch.cat([tst_x_embed, aug_x_embeds], dim=0)
    else:
        augmented_tst_embeds = tst_x_embed
    tst_distribution = (SCALING * augmented_tst_embeds @ tst_prototype).softmax(dim=0).squeeze().cpu().numpy().astype('float16')
    for set_id in range(set_prototypes.size(1)):
        # calculate cost mtx
        retrieved_set_embeds = retrieved_embeds_lst[set_id].to(device)
        set_prototype = set_prototypes[:, set_id]
        A = (augmented_tst_embeds @ set_prototype).unsqueeze(0).repeat_interleave(retrieved_set_embeds.shape[0], dim=0).T
        B = (retrieved_set_embeds @ set_prototype).unsqueeze(0).repeat_interleave(augmented_tst_embeds.shape[0], dim=0)
        C = torch.abs((A-B)).cpu().numpy().astype('float16')
        # reference distribution
        retrieved_distribution = (SCALING * retrieved_set_embeds @ set_prototype).softmax(dim=-1).cpu().numpy().astype('float16')

        import numpy as np

        # Ensure numpy arrays for POT
        if isinstance(tst_distribution, torch.Tensor):
            tst_distribution = tst_distribution.detach().cpu().numpy()
        if isinstance(retrieved_distribution, torch.Tensor):
            retrieved_distribution = retrieved_distribution.detach().cpu().numpy()
        if isinstance(C, torch.Tensor):
            C = C.detach().cpu().numpy()

        # Ensure at least 1-D arrays
        tst_distribution = np.atleast_1d(tst_distribution)
        retrieved_distribution = np.atleast_1d(retrieved_distribution)

        # calculate OT distance
        ot_reg = ot.sinkhorn(tst_distribution, retrieved_distribution, C, reg=cfg['reg'], method='sinkhorn_log', numItermax=100, stopThr=1e-2)
        set_distance_lst.append((np.sum(ot_reg * C)))
    set_relevance = (SCALING * 1/(np.array(set_distance_lst)+1))
    set_relevance = set_relevance.softmax(dim=-1).to(device=device, dtype=torch.float16)
    return id, set_relevance


def compute_set_relevance_score(tst_x_embeds,
                                aug_x_embeds,
                                batch_retrieved_embeds,
                                batch_tst_prototype,
                                batch_set_prototypes,
                                cfg,
                                device):
    batch_results = defaultdict(dict)
    args_lst = []
    batch_id = 0
    args_lst.append((batch_id,
                     tst_x_embeds,
                     aug_x_embeds[batch_id],
                     batch_retrieved_embeds,
                     batch_tst_prototype,
                     batch_set_prototypes,
                     cfg,
                     device))
    batch_id, batch_set_relevance = _compute_ot_based_relevance_score(*args_lst[0])
    batch_results[batch_id] = batch_set_relevance
    batch_results = dict(sorted(batch_results.items()))
    return batch_results


def _compute_retrieval_pred(sample_id, sample_relevance_scores, sample_set_level_pl):
    retrieval_pred = sample_relevance_scores @ sample_set_level_pl
    return sample_id, retrieval_pred


def compute_pred(batch_set_relevance_scores, batch_set_level_pl):
    batch_results = defaultdict(dict)
    args_lst = []
    batch_id = 0
    args_lst.append((batch_id,
                     batch_set_relevance_scores[batch_id],
                     batch_set_level_pl))
    batch_id, batch_retrieval_pred = _compute_retrieval_pred(*args_lst[0])
    batch_results[batch_id] = batch_retrieval_pred
    batch_results = dict(sorted(batch_results.items()))
    return batch_results


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)
        if len(cache_keys) == 0:
            return torch.zeros(1)[0]

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # return alpha * cache_logits, affinity
        return alpha * cache_logits


def get_cache_embeds(cache):
    with torch.no_grad():
        cache_keys = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])

        if len(cache_keys) == 0:
            return torch.zeros(1)[0]

        cache_embeds = torch.cat(cache_keys, dim=0).to(dtype=torch.float32)

        return cache_embeds


def run_test_tda(pos_cfg, neg_cfg, loader, clip_weights, logger, args, cfg):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(loader), [top1], prefix='Test: ')

    with torch.no_grad():

        pos_cache, neg_cache, accuracies = {}, {}, []

        # Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if args.mode in ["clip"]:
            pos_enabled = False
            neg_enabled = False
        if pos_enabled:
            pos_params = {
                k: pos_cfg[k]
                for k in ['shot_capacity', 'alpha', 'beta']
            }
        if neg_enabled:
            neg_params = {
                k: neg_cfg[k]
                for k in [
                    'shot_capacity', 'alpha', 'beta', 'entropy_threshold',
                    'mask_threshold'
                ]
            }

        # Init FAISS GPU resources once
        res = faiss.StandardGpuResources()
        # =========================================================================

        #Test-time adaptation
        for idx, (features, target) in enumerate(loader):
            infer_ori_image = args.datasets in "caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101/ \
            contrast/jpeg/pixelate/elastic/ \
            defocus/glass/zoom/motion/ \
            saturate/ \
            gaussian/shot/impulse/speckle/ \
            fog/frost/snow/brightness"


            tst_x_embeds, aug_x_embeds, clip_logits, loss, prob_map, pred, ori_feat, ori_output = get_clip_logits(
                features,
                clip_weights,
                infer_ori_image=infer_ori_image)

            target, prop_entropy = target.cuda(), get_entropy(
                loss, clip_weights)

            #
            tst_img_desc_sim, aug_img_desc_sim, augmented_desc_sims, aug_x_embeds = get_img_desc_sim(tst_x_embeds, aug_x_embeds, description_encodings, args)

            augmented_dsc_quantile_sims = torch.quantile(augmented_desc_sims.to(dtype=torch.float32), cfg['qthres'],
                                                         dim=1)  # [BS, FD]
            top_k_desc_vals, top_k_desc_ids = augmented_dsc_quantile_sims.topk(k=cfg['top_k_d'], dim=-1)  # [BS, TOPK_D]

            batch_tst_prototypes, batch_set_prototypes, batch_set_pl_results = compute_prototypes(top_k_desc_ids,
                                                                                                  'cuda')

            # No valid text prototypes: fall back to visual cache
            if batch_set_prototypes[0] == None:
                final_logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    if args.mode in ["tda"]:
                        final_logits += compute_cache_logits(tst_x_embeds, pos_cache, pos_params['alpha'],
                                                             pos_params['beta'], clip_weights)
                    elif args.mode in ["mmstta"]:
                        final_logits += compute_cache_logits(tst_x_embeds, cur_pos_cache, pos_params['alpha'],
                                                             pos_params['beta'], clip_weights)

                if neg_enabled and neg_cache:
                    final_logits -= compute_cache_logits(tst_x_embeds, neg_cache, neg_params['alpha'],
                                                         neg_params['beta'], clip_weights, (
                                                         neg_params['mask_threshold']['lower'],
                                                         neg_params['mask_threshold']['upper']))

                adapted_preds = final_logits
                batch_retrieval_preds = clip_logits

                acc = cls_acc(adapted_preds, target)
                accuracies.append(acc)

                top1.update(acc, 1)


                if idx % 10 == 0:
                    progress.display(idx, logger)
                continue

            batch_tst_prototypes = torch.cat(list(batch_tst_prototypes.values()), dim=1)
            batch_set_prototypes = torch.cat(list(batch_set_prototypes.values()), dim=1)
            batch_set_pseudo_labels = torch.cat(list(batch_set_pl_results.values()), dim=0).to(dtype=torch.float16)
            batch_retrieved_embeds = []

            if pos_enabled:
                if args.mode in ["tda", "mmstta"]:
                    update_cache(pos_cache,
                                 pred, [tst_x_embeds, loss, batch_tst_prototypes],
                                 pos_params['shot_capacity'],
                                 fifo=False)
                if args.mode in ["mmstta"]:

                    select_feat, select_output, select_idx = select_confident_samples(ori_feat, ori_output, 0.1)
                    select_entropy = get_output_entropy(select_output)

                    cur_pos_cache = copy.deepcopy(pos_cache)
                    for i in range(select_entropy.shape[0]):
                        cur_pred = int(select_output[i].argmax(dim=-1).item())
                        cur_feat = select_feat[i]
                        update_cache(cur_pos_cache, cur_pred, [cur_feat.unsqueeze(0), select_entropy[i].item()],
                                     pos_params['shot_capacity'] + cfg['delta'], fifo=False)

            if neg_enabled and neg_params['entropy_threshold'][
                    'lower'] < prop_entropy < neg_params['entropy_threshold'][
                        'upper']:
                update_cache(neg_cache, pred,
                             [tst_x_embeds, loss, prob_map],
                             neg_params['shot_capacity'], True)

            if args.mode in ["tda", "mmstta"]:
                cache_img_embeds = get_cache_embeds(pos_cache)

            if not cache_img_embeds.is_cuda:
                cache_img_embeds = cache_img_embeds.to('cuda')

            dimension = cache_img_embeds.shape[1]
            # Reuse FAISS GPU resources
            index_cpu = faiss.IndexFlatL2(dimension)
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            # ====================================================================

            # GPU tensor path via faiss.contrib.torch_utils
            if cache_img_embeds.dtype != torch.float32:
                cache_img_embeds = cache_img_embeds.float()

            if not cache_img_embeds.is_contiguous():
                cache_img_embeds = cache_img_embeds.contiguous()

            index.add(cache_img_embeds)

            current_db_size = index.ntotal

            actual_k = min(cfg['top_k_s'], current_db_size)

            # Build cache-based query prototypes

            # batch_set_pseudo_labels: [Num_Prototypes, Num_Classes] (One-Hot)
            proto_class_ids = batch_set_pseudo_labels.argmax(dim=1)

            cache_based_prototypes_list = []

            for i, cls_id in enumerate(proto_class_ids):
                cls_id = cls_id.item()

                if cls_id in pos_cache and len(pos_cache[cls_id]) > 0:

                    # fifo=False => sorted by loss ascending
                    best_item = pos_cache[cls_id][0]

                    if len(best_item) > 2:
                        best_feat = best_item[2]
                    else:
                        best_feat = best_item[0]

                    best_feat = best_feat.view(-1)
                    best_feat = best_feat / best_feat.norm()

                    cache_based_prototypes_list.append(best_feat)
                else:
                    # Fallback to text prototype if class not cached
                    cache_based_prototypes_list.append(batch_set_prototypes[:, i])

            batch_cache_prototypes = torch.stack(cache_based_prototypes_list, dim=1)

            if torch.cuda.is_available():
                batch_cache_prototypes = batch_cache_prototypes.to('cuda')


            _, top_aux_ids = index.search(batch_cache_prototypes.T.float().contiguous(), actual_k)

            batch_retrieved_embeds = [cache_img_embeds[ids].half() for ids in top_aux_ids]

            batch_set_relevance_scores = compute_set_relevance_score(tst_x_embeds,
                                                                     aug_x_embeds,
                                                                     batch_retrieved_embeds,
                                                                     batch_tst_prototypes,
                                                                     batch_cache_prototypes,
                                                                     cfg,
                                                                     'cuda')

            # Retrieval-based prediction
            batch_retrieval_preds = compute_pred(batch_set_relevance_scores, batch_set_pseudo_labels)
            batch_retrieval_preds = torch.stack(list(batch_retrieval_preds.values()), dim=0)
            batch_retrieval_entp = (-1) * (batch_retrieval_preds * torch.log((batch_retrieval_preds + 1e-7))).sum(
                dim=-1)

            # mode 1
            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                if args.mode in ["tda"]:
                    final_logits += compute_cache_logits(tst_x_embeds, pos_cache, pos_params['alpha'],
                                                         pos_params['beta'], clip_weights)
                elif args.mode in ["mmstta"]:
                    final_logits += compute_cache_logits(tst_x_embeds, cur_pos_cache, pos_params['alpha'],
                                                         pos_params['beta'], clip_weights)

            #
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(tst_x_embeds, neg_cache, neg_params['alpha'],
                                                     neg_params['beta'], clip_weights, (
                                                         neg_params['mask_threshold']['lower'],
                                                         neg_params['mask_threshold']['upper']))

            visual_preds = final_logits.softmax(1)
            visual_entp = softmax_entropy(final_logits)

            # mode 2
            semantic_preds = batch_retrieval_preds
            semantic_entp = batch_retrieval_entp

            # mode 3
            clip_preds = prob_map.clone()
            clip_entp = loss.clone()

            # combine
            if pos_enabled and pos_cache:
                if args.mode in ["tda", "mmstta"]:
                    eps = 1e-7
                    
                    # 1. Instance-level consensus anchor
                    # P_cons = 1/3 * (P_clip + P_vis + P_txt)
                    p_cons = (clip_preds + visual_preds + semantic_preds) / 3.0
                    log_p_cons = torch.log(p_cons + eps)

                    # 2. KL divergence penalty
                    # P_i = D_KL(P_i || P_cons) = sum(P_i * log(P_i / P_cons))
                    kl_clip = (clip_preds * (torch.log(clip_preds + eps) - log_p_cons)).sum(dim=-1)
                    kl_visual = (visual_preds * (torch.log(visual_preds + eps) - log_p_cons)).sum(dim=-1)
                    kl_semantic = (semantic_preds * (torch.log(semantic_preds + eps) - log_p_cons)).sum(dim=-1)

                    # 3. Combined uncertainty
                    # U_i = H_i + P_i 
                    u_clip = clip_entp + kl_clip
                    u_visual = visual_entp + kl_visual
                    u_semantic = semantic_entp + kl_semantic

                    # 4. Consistency-aware gate weights
                    neg_uncertainties = torch.stack([
                        -u_clip,
                        -u_visual,
                        -u_semantic
                    ], dim=0)  # shape: [3, batch_size]

                    weights = F.softmax(neg_uncertainties / args.tau_gate, dim=0)

                    weight_clip = weights[0, :].unsqueeze(1)
                    weight_visual = weights[1, :].unsqueeze(1)
                    weight_semantic = weights[2, :].unsqueeze(1)

                    # 5. Final calibrated prediction
                    combined_preds = (
                            clip_preds * weight_clip +
                            visual_preds * weight_visual +
                            semantic_preds * weight_semantic
                    )
            
            else:
                combined_preds = clip_preds

            acc = cls_acc(combined_preds, target)
            accuracies.append(acc)

            top1.update(acc, 1)


            if idx % 10 == 0:
                progress.display(idx, logger)

        progress.display_summary(logger)

        return sum(accuracies) / len(accuracies)


CLASSNAMES = None
description_embeddings = None
description_encodings = None
NAMES = None
SCALING = None

def main():
    torch.set_num_threads(8)
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    model_path = args.backbone
    clip_model, preprocess = clip.load(model_path)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    fmt = "[{time: MM-DD hh:mm:ss}] {message}"
    config = {
        "handlers": [
            {
                "sink": sys.stderr,
                "format": fmt
            },
        ],
    }
    logger.configure(**config)

    if "search" in args.exp_name:
        file_root = f"log/search/{args.datasets}/"
    else:
        file_root = f"log/eval/{args.datasets}/"
    file_path = f"{file_root}/{args.exp_name}.txt"
    if not os.path.exists(file_root):
        os.makedirs(file_root)
    if os.path.exists(file_path):
        print("Experiment exists. Skipping...")
        exit()
    open(file_path, 'w').close()
    logger.add(file_path, format=fmt)

    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        logger.info(f"Processing {dataset_name} dataset.")

        cfg = get_config_file(config_path, dataset_name)
        logger.info(args)
        logger.info("\nRunning dataset configurations:")
        logger.info(cfg)

        features_path = os.path.join(args.features_root, f'{dataset_name}.pth')

        print(f"Loading payload from {features_path}...")
        features, labels, classnames, template = load_feature_dataset(features_path)

        print(f"Unpacked over！")
        print(f"Class number: {len(classnames)}")
        print(f"Features shape: {features.shape}")

        feature_wrapper = FeatureDataset(features, labels)
        test_loader = DataLoader(
            feature_wrapper,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        global CLASSNAMES
        CLASSNAMES = [i.replace('_', ' ') for i in classnames]

        clip_weights = clip_classifier(CLASSNAMES, template, clip_model)

        # description embeddings
        global description_encodings
        cupl_prompts = json.load(open(args.cupl_root.format(dataset_name)))
        description_encodings = defaultdict()

        pre_template = get_pre_tempalte(dataset_name)

        global description_embeddings
        with torch.no_grad():
            for c_idx, c_name in enumerate(CLASSNAMES):
                if c_name in cupl_prompts:
                    prompts_list = cupl_prompts[c_name]
                elif c_name.lower() in cupl_prompts:
                    prompts_list = cupl_prompts[c_name.lower()]
                else:
                    print(f"Error: Class '{c_name}' not found in cupl_prompts keys.")
                    raise KeyError(c_name)
                v = [pre_template.format(c_name.replace('_', ' ')) + v_i for v_i in prompts_list]
                v.append(pre_template.format(c_name.replace('_', ' '))[:-1])
                tokens = clip.tokenize(v, truncate=True).cuda()
                description_encodings[c_idx] = F.normalize(clip_model.encode_text(tokens))
            description_embeddings = torch.cat(list(description_encodings.values()))

        global NAMES
        NAMES = []
        for c_idx, c_name in enumerate(CLASSNAMES):
            NAMES += [int(c_idx)] * (len(cupl_prompts[c_name]) + 1)
        NAMES = torch.Tensor(NAMES).to(device='cuda', dtype=torch.int16)

        global SCALING
        SCALING = clip_model.logit_scale.exp().cpu()

        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_weights, logger, args, cfg)


if __name__ == "__main__":
    main()
