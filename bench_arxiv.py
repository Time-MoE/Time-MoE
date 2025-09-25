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


def get_argparser():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Train a configurable MLP with cosine LR scheduler and W&B logging"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/share/dean/arxiv-data/model_dev/baseline_benchmarking",
        help="data root",
    )
    parser.add_argument(
        "--input_horizon", type=int, default=365, help="Input feature size"
    )
    parser.add_argument("--output_size", type=int, default=1, help="Output size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--ablation", default=False, action="store_true")
    parser.add_argument("--ablation_name", default="accesses", type=str)
    parser.add_argument(
        "--output_dir", default="/share/dean/embeddings_large_arxiv_accesses", type=str
    )
    return parser


def get_datasets(
    input_horizon,
    root="baseline_benchmarking",
    train_name="train",
    val_name="val",
    test_name="labeled_test",
    ablation=False,
    ablation_name="accesses",
):
    if ablation:
        train_dataset = ArxivDatasetAblation(
            root, train_name, input_horizon, ablation_name=ablation_name
        )
        val_dataset = ArxivDatasetAblation(
            root, val_name, input_horizon, ablation_name=ablation_name
        )
        test_dataset = ArxivDatasetAblation(
            root, test_name, input_horizon, ablation_name=ablation_name
        )
    else:
        train_dataset = ArxivDataset(root, train_name, input_horizon)
        val_dataset = ArxivDataset(root, val_name, input_horizon)
        test_dataset = ArxivDataset(root, test_name, input_horizon)
    return train_dataset, val_dataset, test_dataset


def main(args):
    if args.ablation:
        args.input_size = args.input_horizon
    else:
        args.input_size = args.input_horizon * 2

    train_dataset, val_dataset, test_dataset = get_datasets(
        args.input_horizon,
        args.data_root,
        ablation=args.ablation,
        ablation_name=args.ablation_name,
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
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    for loader, path in [(train_loader, train_path), (test_loader, test_path)]:
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
