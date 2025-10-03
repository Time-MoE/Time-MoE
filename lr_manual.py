import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score
from datasets import load_from_disk
import pandas as pd
import wandb
from tqdm import tqdm
import os


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def logistic_regression_pytorch(
    train_dataset,
    test_dataset,
    feature_col,
    meta_data={},
    threshold=50,
    batch_size=128,
    epochs=100,
    lr=0.01,
    device="cuda",
):
    """
    PyTorch logistic regression evaluation with DataLoader.

    Args:
        train_dataset: HF dataset for training
        test_dataset: HF dataset for testing
        feature_col: column name for features (e.g. 'mid_hs_mean')
        threshold: threshold for binary labels
        batch_size: batch size for DataLoader
        epochs: number of training epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
    """

    X_train = torch.FloatTensor(np.array(train_dataset[feature_col]))
    X_test = torch.FloatTensor(np.array(test_dataset[feature_col]))

    # Create binary labels
    y_train = torch.FloatTensor(
        (np.array(train_dataset["orig_labels"]) > threshold).astype(float)
    )
    y_test = torch.FloatTensor(
        (np.array(test_dataset["orig_labels"]) > threshold).astype(float)
    )
    print("loaded data")

    # Create DataLoaders directly - no reshaping needed!
    train_dataset_torch = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset_torch, batch_size=batch_size, shuffle=True)

    test_dataset_torch = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset_torch, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X_train.shape[1]
    model = LogisticRegression(input_size)

    model = model.cuda()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    print("Starting training...")
    wandb.init(project="logistic_regression", name=f"{feature_col}_{threshold}")
    model.train()
    step = 0
    for epoch in range(epochs):
        total_loss = 0
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_X, batch_y in train_loader:
            if device == "cuda":
                batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1
            # progress_bar.set_postfix(loss=loss.item())
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/step": step})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")
        wandb.log({"train/avg_epoch_loss": avg_loss, "train/epoch": epoch + 1})

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            test_metrics = eval(model, train_loader, test_loader, y_train, y_test)
            test_metrics.update({"test/epoch": epoch + 20})

            wandb.log(test_metrics)
    wandb.log(meta_data)
    wandb.finish()
    return test_metrics


def eval(model, train_loader, test_loader, y_train, y_test, device="cuda"):
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Get training predictions
        train_probs = []
        train_preds = []
        for batch_X, _ in train_loader:
            batch_X = batch_X.cuda()
            outputs = model(batch_X).squeeze()
            probs = outputs.cpu().numpy()
            train_probs.extend(probs)
            train_preds.extend((probs > 0.5).astype(int))

        # Get test predictions
        test_probs = []
        test_preds = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.cuda()
            outputs = model(batch_X).squeeze()
            probs = outputs.cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend((probs > 0.5).astype(int))

    # Calculate metrics
    return {
        "train_auc": float(roc_auc_score(y_train, train_probs)),
        "val_auc": float(roc_auc_score(y_test, test_probs)),
        "train_f1": float(f1_score(y_train, train_preds)),
        "val_f1": float(f1_score(y_test, test_preds)),
    }


# Main execution
path = "/share/dean/embeds/embed"
results_list = []

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test different feature types

case = "github_pusheswatches"
threshold = 10
ih = 30
path = f"/share/dean/embeds/embed_{case}"
train_dataset = load_from_disk(path + f"_{case}_train_ds")
test_dataset = load_from_disk(path + f"_{case}_test_ds")
print(f"\nTraining for {case} dataset:")

for feature in [
    "mid_hs_mean",
    "final_hs_mean",
    "mid_hs_max",
    "final_hs_max",
    "mid_hs_last",
    "final_hs_last",
]:
    meta_data = {
        "feature": feature,
        "horizon": ih,
        "dataset": case,
    }
    print(f"  Feature: {feature}")
    results = logistic_regression_pytorch(
        train_dataset,
        test_dataset,
        feature,
        meta_data=meta_data,
        threshold=threshold,
        batch_size=128,
        epochs=100,
        lr=0.001,
        device=device,
    )
    print(f"    Results: {results}")
    row = {
        "dataset": case,
        "feature": feature,
        "threshold": threshold,
        "horizon": ih,
        **results,
    }
    results_list.append(row)

df = pd.DataFrame(results_list)
df.to_csv(f"pytorch_logistic_regression_results_{case}_{ih}.csv", index=False)

print(f"\nSaved {len(results_list)} results to pytorch_logistic_regression_results.csv")
