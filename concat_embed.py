import numpy as np
import os
from datasets import Dataset
from glob import glob
import argparse


def create_dataset_from_batches(path, batch_count):
    """
    Create HF dataset from saved batch files.
    Each row represents one sample from the batches.
    """

    def batch_generator():
        """Generator that yields one row at a time for memory efficiency"""
        for batch_idx in range(batch_count):
            # Load all arrays for this batch
            mid_hs_mean = np.load(f"{path}/mid_hs_mean_{batch_idx}.npy")
            final_hs_mean = np.load(f"{path}/final_hs_mean_{batch_idx}.npy")
            mid_hs_max = np.load(f"{path}/mid_hs_max_{batch_idx}.npy")
            final_hs_max = np.load(f"{path}/final_hs_max_{batch_idx}.npy")
            mid_hs_last = np.load(f"{path}/mid_hs_last_{batch_idx}.npy")
            final_hs_last = np.load(f"{path}/final_hs_last_{batch_idx}.npy")
            orig_labels = np.load(f"{path}/orig_labels_{batch_idx}.npy")

            # Yield each sample in the batch as a separate row
            batch_size = len(mid_hs_mean)
            for i in range(batch_size):
                yield {
                    "mid_hs_mean": mid_hs_mean[i],
                    "final_hs_mean": final_hs_mean[i],
                    "mid_hs_max": mid_hs_max[i],
                    "final_hs_max": final_hs_max[i],
                    "mid_hs_last": mid_hs_last[i],
                    "final_hs_last": final_hs_last[i],
                    "orig_labels": orig_labels[i],
                }

    return Dataset.from_generator(batch_generator)


def create_dataset_auto(path):
    """Auto-detect number of batches from files"""
    batch_files = glob(f"{path}/mid_hs_mean_*.npy")
    batch_count = len(batch_files)
    return create_dataset_from_batches(path, batch_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/share/dean/embeds/")

    args = parser.parse_args()
    train_dataset = create_dataset_auto(args.path + "_train")
    train_dataset.save_to_disk(args.path + "_train_ds")
    test_dataset = create_dataset_auto(args.path + "_test")
    test_dataset.save_to_disk(args.path + "_test_ds")
