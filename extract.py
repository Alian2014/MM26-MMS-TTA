import torch
import clip
import os
import argparse
from tqdm import tqdm
import random

from utils import *
from utils import AverageMeter

def extract_and_pack(args):
    # Limit PyTorch CPU threads
    torch.set_num_threads(8)
    # Initialize CLIP model
    model_path = args.backbone
    clip_model, preprocess = clip.load(model_path)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    print("Building original data loader...")

    # Support multiple datasets separated by /
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        orig_loader, classnames, template = build_test_data_loader(
            dataset_name,
            args.data_root,
            preprocess,
            args
        )

        print(f"Start extracting features for {dataset_name}...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        all_features = []
        all_labels = []

        for idx, (images, target) in enumerate(tqdm(orig_loader)):
            with torch.no_grad():
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).cuda()
                else:
                    images = images.cuda()

                # CLIP encode [B, D]
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Convert to float16 and move to CPU to save memory
                feat_tensor = image_features.cpu().half()

                label = target.item() if isinstance(target, torch.Tensor) else target

                all_features.append(feat_tensor)
                all_labels.append(label)

        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(f"[{dataset_name}] Peak Memory Cost: {peak_mem_gb:.2f} GB")
        print("Stacking tensors (this may take a moment)...")

        # Stack to contiguous tensor: [Total_Samples, Views, Dim]
        final_features = torch.stack(all_features)

        final_labels = torch.tensor(all_labels)

        print(f"Packed Shape: Features {final_features.shape}, Labels {final_labels.shape}")

        payload = {
            "classnames": classnames,
            "template": template,
            "features": final_features,
            "labels": final_labels
        }

        full_path = os.path.join(args.save_path, f'{dataset_name}.pth')
        os.makedirs(args.save_path, exist_ok=True)
        torch.save(payload, full_path)
        print(f"Successfully packed features to {full_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default='./features/vit_b16',
                        help='Path to save feature files. Default is ./features/vit_b16')

    parser.add_argument(
        '--datasets',
        dest='datasets',
        type=str,
        required=True,
        help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S"
    )
    parser.add_argument(
        '--data-root',
        dest='data_root',
        type=str,
        default='/path/to/datasets',
        help='Path to the datasets directory. Default is /path/to/datasets')
    parser.add_argument('--backbone',
                        dest='backbone',
                        type=str,
                        choices=['RN50', 'ViT-B/16'],
                        required=True,
                        help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--views', type=int, default=64)

    args = parser.parse_args()
    extract_and_pack(args)
