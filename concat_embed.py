import numpy as np
import os
from datasets import Dataset
from glob import glob
import argparse


def create_dataset_from_batches(path, batch_count, cleanup=True):
    """
    Create HF dataset from saved batch files.
    Each row represents one sample from the batches.

    Args:
        path: Path to the batch files
        batch_count: Number of batches to process
        cleanup: If True, delete source .npy files after processing each batch
    """

    def batch_generator():
        """Generator that yields one row at a time for memory efficiency"""
        for batch_idx in range(batch_count):
            print(f"Processing batch {batch_idx + 1}/{batch_count}")

            # Define file paths for this batch
            files_to_load = [
                f"{path}/mid_hs_mean_{batch_idx}.npy",
                f"{path}/final_hs_mean_{batch_idx}.npy",
                f"{path}/mid_hs_max_{batch_idx}.npy",
                f"{path}/final_hs_max_{batch_idx}.npy",
                f"{path}/mid_hs_last_{batch_idx}.npy",
                f"{path}/final_hs_last_{batch_idx}.npy",
                f"{path}/orig_labels_{batch_idx}.npy",
            ]

            # Load all arrays for this batch
            mid_hs_mean = np.load(files_to_load[0])
            final_hs_mean = np.load(files_to_load[1])
            mid_hs_max = np.load(files_to_load[2])
            final_hs_max = np.load(files_to_load[3])
            mid_hs_last = np.load(files_to_load[4])
            final_hs_last = np.load(files_to_load[5])
            orig_labels = np.load(files_to_load[6])

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

            # Clean up: delete the source files after processing this batch
            if cleanup:
                print(f"Cleaning up batch {batch_idx} files...")
                for file_path in files_to_load:
                    if os.path.exists(file_path):
                        os.remove(file_path)

    return Dataset.from_generator(batch_generator)


def create_dataset_auto(path, cleanup=True):
    """Auto-detect number of batches from files"""
    batch_files = glob(f"{path}/mid_hs_mean_*.npy")
    batch_count = len(batch_files)
    print(f"Found {batch_count} batches to process")
    return create_dataset_from_batches(path, batch_count, cleanup=cleanup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/share/dean/embeds/")
    parser.add_argument(
        "--cleanup",  # Changed from --no-cleanup
        action="store_true",
        help="Delete source .npy files after processing",
    )

    args = parser.parse_args()
    cleanup = args.cleanup
    if cleanup:
        print("WARNING: Source .npy files will be deleted as they are processed!")
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            exit(1)

    print("Processing training dataset...")
    train_dataset = create_dataset_auto(args.path + "_train", cleanup=cleanup)
    train_dataset.save_to_disk(args.path + "_train_ds")
    print("Training dataset saved!")

    print("Processing test dataset...")
    test_dataset = create_dataset_auto(args.path + "_test", cleanup=cleanup)
    test_dataset.save_to_disk(args.path + "_test_ds")
    print("Test dataset saved!")

    print("All done!")
