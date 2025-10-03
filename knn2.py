import numpy as np
import faiss
from datasets import load_from_disk
import pandas as pd
import argparse
# export PYTHONPATH=~/faiss_xx/build/faiss/python/build/lib:$PYTHONPATH


def get_mae(preds, gts):
    """Calculate mean absolute error and standard deviation."""
    absolute_errors = np.abs(preds - gts)
    return np.mean(absolute_errors), np.std(absolute_errors)


def get_relative_mae(preds, gts, thresh):
    """Calculate relative MAE for predictions above threshold."""
    mask = gts >= thresh
    print(
        "Number of positive samples:",
        np.sum(mask),
        "out of",
        len(gts),
        f"thresh={thresh}    ",
    )
    if np.sum(mask) == 0:
        return np.nan, np.nan
    absolute_errors = np.abs(preds - gts)
    relative_errors = absolute_errors[mask] / gts[mask]
    return np.mean(relative_errors), np.std(relative_errors)


# Main execution
base_path = "/share/dean/embeds/"
k = 5
results_list = []
ihs = {30, 100, 365}  # input horizon values

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case", type=str, default="arxiv", help="Dataset case: arxiv or github"
)
args = parser.parse_args()
case = args.case
if case == "arxiv":
    relative_thresh = 50
elif case == "github":
    relative_thresh = 10
else:
    raise ValueError("Invalid case. Choose 'arxiv' or 'github'.")

for ih in ihs:
    if ih == 365:
        path = f"{base_path}embed_{case}"
    else:
        path = f"{base_path}embed_{case}_{ih}"

    train_dataset = load_from_disk(path + "_train_ds")
    test_dataset = load_from_disk(path + "_test_ds")
    print(f"\nResults for {case} dataset (k={k}, input horizon={ih}):")

    labels = np.array(train_dataset["orig_labels"]).astype(np.float32)

    actual = np.array(test_dataset["orig_labels"]).astype(np.float32)

    for feature in [
        "mid_hs_mean",
        "final_hs_mean",
        "mid_hs_max",
        "final_hs_max",
        "mid_hs_last",
        "final_hs_last",
    ]:
        print(f"  Processing {feature}...")

        # Prepare knowledge base and test base
        knowledge_base = np.array(train_dataset[feature]).astype(np.float32)
        test_base = np.array(test_dataset[feature]).astype(np.float32)

        assert knowledge_base.dtype == np.float32
        assert test_base.dtype == np.float32

        # Get dimension
        d = knowledge_base.shape[1]

        # Setup FAISS
        res = faiss.StandardGpuResources()
        index_cpu = faiss.IndexFlatL2(d)
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

        # Add to index and search
        index_gpu.add(knowledge_base)  # Adding to GPU index
        _, I = index_gpu.search(test_base, k)  # Searching on GPU

        # Make predictions
        pred = []
        temp = []
        for neighbor_row in I:
            for idx in neighbor_row:
                temp.append(labels[idx])
            temp = np.array(temp)
            median_vector = np.median(temp, axis=0)
            pred.append(median_vector)
            temp = []
        pred = np.array(pred)

        # Get actual values
        # actual = np.array(test_dataset["orig_labels"]).astype(np.float32)

        # Calculate metrics
        mae_log = get_mae(np.log1p(pred), np.log1p(actual))[0]
        mae_raw = get_mae(pred.round(), actual)[0].round(3)
        rel_mae = get_relative_mae(pred.round(), actual, relative_thresh)[0].round(3)

        results = {
            "mae_log": float(mae_log),
            "mae_raw": float(mae_raw),
            "relative_mae": float(rel_mae) if not np.isnan(rel_mae) else None,
        }

        row = {
            "dataset": case,
            "horizon": ih,  # Add this line
            "feature": feature,
            "k": k,
            "relative_thresh": relative_thresh,
            **results,
        }
        results_list.append(row)

        print(f"    MAE (log): {results['mae_log']:.3f}")
        print(f"    MAE (raw): {results['mae_raw']:.3f}")
        print(f"    Relative MAE: {results['relative_mae']:.3f}")

# Save results
df = pd.DataFrame(results_list)
df.to_csv(f"faiss_knn_results_{case}.csv", index=False)
print(f"\nSaved {len(results_list)} results to faiss_knn_results_{case}.csv")
