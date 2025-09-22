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
    parser.add_argument("--ablation_name", default="accesses", type=str)
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    model = TimeMoeForPrediction.from_pretrained(
        "Maple728/TimeMoE-200M",
        device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        trust_remote_code=True,
    )
    prediction_length = 1
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(train_loader)):
            inputs, labels, orig_labels = batch
            inputs = inputs.to("cuda")
            hidden_states = model.generate(inputs, max_new_tokens=prediction_length)
            # filename = f"/share/dean/embeddings_large_arxiv_train/output_embeddings_{self.batch_id}.npy"
            np.save(orig_labels.cpu().numpy(), f"labels_{idx}.npy")
        print(idx, "batches processed")


if __name__ == "__main__":
    args = get_argparser().parse_args()
    main(args)
