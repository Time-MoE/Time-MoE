import numpy as np
import glob
from datasets import Dataset
import gc
from tqdm import tqdm


def create_embeddings_dataset(embeddings_dir: str, strategy: str = "mean"):
    """
    Create HuggingFace Dataset where each item is a single embedding.
    Each of the 128 embeddings from each .npy file becomes one dataset item.
    """
    embedding_files = sorted(glob.glob(f"{embeddings_dir}/output_embeddings_*.npy"))

    def generate_individual_embeddings():
        """Generator that yields one embedding at a time"""
        global_idx = 0

        for file_idx, file_path in enumerate(tqdm(embedding_files)):
            print(f"Processing file {file_idx + 1}/{len(embedding_files)}")

            embeddings = np.load(file_path)  # Shape: (128, 730, 768)

            # Process each of the 128 embeddings individually
            for sample_idx in range(embeddings.shape[0]):  # Loop through 128 samples
                single_embedding = embeddings[sample_idx]  # Shape: (730, 768)

                # Apply pooling strategy to this single embedding
                if strategy == "last":
                    pooled = single_embedding[-1, :]  # Shape: (768,)
                elif strategy == "mean":
                    pooled = single_embedding.mean(axis=0)  # Shape: (768,)
                elif strategy == "max":
                    pooled = single_embedding.max(axis=0)  # Shape: (768,)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                yield {
                    "id": global_idx,
                    "batch_idx": file_idx,
                    "sample_idx": sample_idx,
                    "embedding": pooled.tolist(),  # Convert to list for serialization
                }
                global_idx += 1

            # Clean up after each file
            del embeddings
            gc.collect()

    # Create dataset from generator
    dataset = Dataset.from_generator(generate_individual_embeddings)
    return dataset


# Usage example:
if __name__ == "__main__":
    train_path = "/share/dean/embeddings_large_arxiv_train"
    test_path = "/share/dean/embeddings_large_arxiv"
    for path in [train_path, test_path]:
        print(f"Creating datasets for path: {path}")
        # Create dataset where each item is one embedding
        for strategy in ["last", "mean", "max"]:
            print(f"Creating dataset with strategy: {strategy}")
            dataset = create_embeddings_dataset(path, strategy=strategy)
            dataset.save_to_disk(f"{train_path}/embed_{strategy}")

            # Use the dataset
            print(f"Dataset size: {len(dataset)}")
            print(
                f"Each file has 128 embeddings, so total = {len(dataset)} individual embeddings"
            )
            print(f"First sample: {dataset[0]}")
            print(f"Embedding shape: {len(dataset[0]['embedding'])}")

            # Access individual embeddings
            first_embedding = np.array(dataset[0]["embedding"])
            print(f"First embedding as numpy array shape: {first_embedding.shape}")

            # Batch access for training
            batch_embeddings = np.array([dataset[i]["embedding"] for i in range(32)])
            print(f"Batch of 32 embeddings shape: {batch_embeddings.shape}")

            # Filter by batch if needed
            batch_0_items = dataset.filter(lambda x: x["batch_idx"] == 0)
            print(f"Items from first batch file: {len(batch_0_items)}")  # Should be 128
