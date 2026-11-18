import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image
import random
from enum import Enum

from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def get_output_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * torch.exp(logits)).sum(dim=-1)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def select_confident_samples(feat, logits, topTPT):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    # return feat[idxTPT], logits[idxTPT]
    return feat[idxTPT], logits[idxTPT], idxTPT


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(
        0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def mask_logits(logits_list, logits_neutral, score_db, classnames, pred_index, n, threshold):
    """
    (Corrected Version for float16)
    Iterates through the logits list. If the attribute fails EITHER check
    (score < n OR difference < threshold), it sets the logit to the
    minimum representable value for the tensor's data type.

    (Changelog: MASK_VALUE is now defined dynamically inside the function)

    :param logits_list: The 1D tensor of attribute logits to modify (e.g., torch.float16).
    :param logits_neutral: The 1D tensor of NEUTRAL attribute logits.
    :param score_db: Your loaded {classname: {0: score, 1: score, ...}} database.
    :param classnames: The complete list of class names.
    :param pred_index: The index of the current class.
    :param n: The score threshold (from score_db).
    :param threshold: The logit difference threshold.
    :return: The modified (masked) logits_list.
    """

    # --- 1. (Integrated) Get and transform class name ---
    try:
        if pred_index >= len(classnames):
            raise IndexError(f"Index {pred_index} is out of bounds for classnames list")
        classname_raw = classnames[pred_index]
        classname_str = classname_raw.replace('_', ' ')
    except Exception as e:
        print(f"Error: Failed to get classname: {e}")
        return logits_list

    # --- 2. Get score dictionary ---
    try:
        scores_for_class = score_db[classname_str]
    except KeyError:
        print(f"Warning: Class '{classname_str}' not found in score_db. Logits were not modified.")
        return logits_list

    # --- 3. Get the single baseline neutral logit ---
    if len(logits_neutral) == 0:
        print(f"Error: The neutral logits list is empty.")
        return logits_list
    baseline_neutral_score = logits_neutral[0]

    # --- 4. (!! FIX !!) Define MASK_VALUE based on the input tensor's type ---
    #
    # This gets the minimum finite value for the tensor's dtype
    # (e.g., -65504.0 for float16, or a huge number for float32/64)
    #
    MASK_VALUE = torch.finfo(logits_list.dtype).min

    # (Alternative simple fix: MASK_VALUE = float('-inf'))
    # (Using .min is often safer as it avoids -inf in calculations)

    # --- 5. Iterate and apply masking logic ---
    for i in range(len(logits_list)):

        # Get Condition 1: Check the pre-computed score
        score = scores_for_class.get(i, 0.0)

        # Get Condition 2: Check the logit difference
        # We must read logits_list[i] *before* potentially masking it
        difference = logits_list[i] - baseline_neutral_score

        # Apply masking logic
        if score < n or difference < threshold:
            # This assignment is now safe (e.g., float16 = float16)
            logits_list[i] = MASK_VALUE

    # 6. Return the modified list
    return logits_list


def get_clip_logits(features,
                    clip_weights,
                    infer_ori_image=False):
    with torch.no_grad():
        image_features = features.squeeze(0).cuda()
        tst_x_embeds = image_features[:1]
        aug_x_embeds = image_features[1:]

        clip_logits = 100. * image_features @ clip_weights
        ori_feat = image_features.detach().clone()
        ori_output = clip_logits.detach().clone()

        if image_features.size(0) > 1:
            if infer_ori_image:
                loss = softmax_entropy(clip_logits[:1])
                prob_map = clip_logits[:1].softmax(1)
                pred = int(clip_logits[:1].topk(1, 1, True, True)[1].t()[0])

                return tst_x_embeds, aug_x_embeds, clip_logits[:
                                                      1], loss, prob_map, pred, ori_feat, ori_output
            else:
                batch_entropy = softmax_entropy(clip_logits)
                selected_idx = torch.argsort(
                    batch_entropy,
                    descending=False)[:int(batch_entropy.size()[0] * 0.1)]
                output = clip_logits[selected_idx]
                image_features = image_features[selected_idx].mean(
                    0).unsqueeze(0)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                clip_logits = output.mean(0).unsqueeze(0)

                loss = avg_entropy(output).unsqueeze(0)
                prob_map = output.softmax(1).mean(0).unsqueeze(0)
                pred = int(
                    output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return tst_x_embeds, aug_x_embeds, clip_logits, loss, prob_map, pred, ori_feat, ori_output


def get_ood_preprocess(args):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)
    ])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    # aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)
    aug_preprocess = AugMixAugmenter(base_transform,
                                     preprocess,
                                     n_views=args.views - 1,
                                     augmix=False)

    return aug_preprocess


def get_cross_dataset_preprocess(preprocess, args):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([])
    # aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)
    aug_preprocess = AugMixAugmenter(base_transform,
                                     preprocess,
                                     n_views=args.views - 1,
                                     augmix=False)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"

    config_file = os.path.join(config_path, config_name)

    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess, args):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test,
                                                  batch_size=1,
                                                  num_workers=4,
                                                  shuffle=True)

    elif dataset_name in ['A', 'V', 'R', 'S']:
        preprocess = get_ood_preprocess(args)
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test,
                                        batch_size=1,
                                        is_train=False,
                                        tfm=preprocess,
                                        shuffle=True)

    elif dataset_name in [
            'caltech101', 'dtd', 'eurosat', 'fgvc', 'food101',
            'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397',
            'ucf101'
    ]:
        preprocess = get_cross_dataset_preprocess(preprocess, args)
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test,
                                        batch_size=1,
                                        is_train=False,
                                        tfm=preprocess,
                                        shuffle=True)

    else:
        raise "Dataset is not from the chosen list"
    return test_loader, dataset.classnames, dataset.template


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def display_summary(self, logger):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features: Tensor [N, 64, 512], dtype=torch.float16
            labels: Tensor [N], dtype=torch.long
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]  # [64, 512], float16
        label = self.labels[idx]
        # feat = feat.float()

        return feat, label


def load_feature_dataset(pth_path):
    """Load stacked feature tensors and metadata from a .pth file."""
    print(f"Loading features from {pth_path} ...")

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"File not found: {pth_path}")

    payload = torch.load(pth_path, map_location='cpu')

    features = payload['features']  # Expect: [N, 64, 512] (float16)
    labels = payload['labels']  # Expect: [N]
    classnames = payload['classnames']
    template = payload['template']

    print(f"Dataset Loaded.")
    print(f"  - Shape: {features.shape}")
    print(f"  - Dtype: {features.dtype} (Should be float16)")
    print(f"  - Memory: {features.element_size() * features.nelement() / 1024 ** 3:.2f} GB")

    return features, labels, classnames, template
