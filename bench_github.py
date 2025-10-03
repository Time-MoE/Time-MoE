from baseline_datasets import (
    ArxivDataset,
    ArxivDatasetAblation,
    GitHubDataset,
    GitHubDatasetAblation,
)
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil


def get_argparser():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Train a configurable MLP with cosine LR scheduler and W&B logging"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/share/dean/arxiv-data/model_dev/github_baseline_benchmarking",
        help="data root",
    )
    parser.add_argument(
        "--input_horizon", type=int, default=365, help="Input feature size"
    )
    parser.add_argument("--output_size", type=int, default=1, help="Output size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--ablation", default=False, action="store_true")
    parser.add_argument(
        "--ablation_name",
        type=str,
        nargs="+",  # accepts one or more values
        default=["forks"],  # default as a list
        help="Ablation feature(s) to use",
    )
    parser.add_argument(
        "--output_dir", default="/share/dean/embeddings_large_github_pushes", type=str
    )
    return parser


def get_datasets(
    input_horizon,
    root="baseline_benchmarking",
    train_name="train",
    val_name="val",
    test_name="test",
    ablation=False,
    ablation_name="pushes",
):
    if ablation:
        train_dataset = GitHubDatasetAblation(
            root, train_name, input_horizon, ablation_name
        )
        val_dataset = GitHubDatasetAblation(
            root, val_name, input_horizon, ablation_name
        )
        test_dataset = GitHubDatasetAblation(
            root, test_name, input_horizon, ablation_name
        )
    else:
        train_dataset = GitHubDataset(root, train_name, input_horizon)
        val_dataset = GitHubDataset(root, val_name, input_horizon)
        test_dataset = GitHubDataset(root, test_name, input_horizon)
    return train_dataset, val_dataset, test_dataset


def main(args):
    ablation_name = args.ablation_name

    if args.ablation:
        args.input_size = args.input_horizon * len(ablation_name)
    else:
        args.input_size = args.input_horizon * 3

    train_dataset, val_dataset, test_dataset = get_datasets(
        args.input_horizon,
        args.data_root,
        ablation=args.ablation,
        ablation_name=ablation_name,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TimeMoeForPrediction.from_pretrained(
        "Maple728/TimeMoE-200M",
        device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        trust_remote_code=True,
    )
    train_path = args.output_dir + "_train"
    test_path = args.output_dir + "_test"

    print("CAUTION: Deleting existing embedding directories if they exist!")
    if os.path.exists(train_path):
        shutil.rmtree(train_path, ignore_errors=True)
    if os.path.exists(test_path):
        shutil.rmtree(test_path, ignore_errors=True)

    os.makedirs(train_path)
    os.makedirs(test_path)

    for loader, path in [(train_loader, train_path), (test_loader, test_path)]:
        # loader = test_loader
        # path = test_path
        print(f"Generating embeddings and saving to {path}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                inputs, labels, orig_labels = batch
                inputs = inputs.to("cuda")
                mid_hs, final_hs = model(inputs)
                np.save(f"{path}/mid_hs_mean_{batch_idx}.npy", mid_hs.mean(axis=1))
                np.save(f"{path}/final_hs_mean_{batch_idx}.npy", final_hs.mean(axis=1))
                np.save(f"{path}/mid_hs_max_{batch_idx}.npy", mid_hs.max(axis=1))
                np.save(f"{path}/final_hs_max_{batch_idx}.npy", final_hs.max(axis=1))
                np.save(f"{path}/mid_hs_last_{batch_idx}.npy", mid_hs[:, -1, :])
                np.save(f"{path}/final_hs_last_{batch_idx}.npy", final_hs[:, -1, :])
                np.save(
                    f"{path}/orig_labels_{batch_idx}.npy",
                    orig_labels.cpu().numpy(),
                )
            print(batch_idx + 1, "batches processed")


if __name__ == "__main__":
    args = get_argparser().parse_args()
    main(args)
