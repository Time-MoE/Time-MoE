from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import os.path as osp
import torch
import numpy as np


class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Compute mean and std from training data"""
        self.mean = X.mean()
        self.std = X.std()
        return self.mean, self.std

    def transform(self, X):
        """Standardize the data"""
        mean, std = self.mean, self.std
        return (X - mean) / (std + 1e-6)

    def inverse_transform(self, X_scaled):
        """Revert to original scale"""
        mean, std = self.mean, self.std
        return X_scaled * (std + 1e-6) + mean


class ArxivDataset(Dataset):
    def __init__(self, root, split, input_horizon):
        self.root = root
        self.split = split
        ds = load_from_disk(osp.join(root, split))
        scaler = TorchStandardScaler()
        self.citations = ds["citations_input"][:, :input_horizon]
        self.accesses = ds["accesses_input"][:, :input_horizon]
        self.orig_labels = ds["citations_label"]
        self.citations = torch.log(1.0 + torch.from_numpy(self.citations).float())
        self.accesses = torch.log(1.0 + torch.from_numpy(self.accesses).float())
        self.labels = torch.log(1.0 + torch.from_numpy(self.orig_labels).float())
        self.citation_mean, self.citation_std = scaler.fit(self.citations)
        self.access_mean, self.access_std = scaler.fit(self.accesses)
        self.citations = (self.citations - self.citation_mean) / (
            self.citation_std + 1e-6
        )
        self.accesses = (self.accesses - self.access_mean) / (self.access_std + 1e-6)
        self.labels_mean, self.labels_std = scaler.fit(self.labels)
        self.labels = (self.labels - self.labels_mean) / (self.labels_std + 1e-6)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(
            np.concatenate([self.citations[idx], self.accesses[idx]], axis=0)
        )
        orig_label = torch.tensor(self.orig_labels[idx]).float()
        label = self.labels[idx]
        return inputs, label, orig_label


class GitHubDataset(Dataset):
    def __init__(self, root, split, input_horizon):
        self.root = root
        self.split = split
        ds = load_from_disk(osp.join(root, split))
        self.forks = ds["forks_inputs"][:, :input_horizon]
        self.stars = ds["watches_inputs"][:, :input_horizon]
        self.pushes = ds["pushes_inputs"][:, :input_horizon]
        self.orig_labels = ds["forks_labels"]
        self.forks = torch.log(1.0 + torch.from_numpy(self.forks).float())
        self.stars = torch.log(1.0 + torch.from_numpy(self.stars).float())
        self.pushes = torch.log(1.0 + torch.from_numpy(self.pushes).float())
        self.labels = torch.log(1.0 + torch.from_numpy(self.orig_labels).float())
        scaler = TorchStandardScaler()
        self.fork_mean, self.fork_std = scaler.fit(self.forks)
        self.star_mean, self.star_std = scaler.fit(self.stars)
        self.push_mean, self.push_std = scaler.fit(self.pushes)
        self.forks = (self.forks - self.fork_mean) / (self.fork_std + 1e-6)
        self.stars = (self.stars - self.star_mean) / (self.star_std + 1e-6)
        self.pushes = (self.pushes - self.push_mean) / (self.push_std + 1e-6)
        self.labels_mean, self.labels_std = scaler.fit(self.labels)
        self.labels = (self.labels - self.labels_mean) / (self.labels_std + 1e-6)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(
            np.concatenate([self.forks[idx], self.stars[idx], self.pushes[idx]], axis=0)
        )
        orig_label = torch.tensor(self.orig_labels[idx]).float()
        label = self.labels[idx]
        return inputs, label, orig_label


class ArxivDatasetAblation(Dataset):
    def __init__(self, root, split, input_horizon, ablation_name="accesses"):
        self.root = root
        self.split = split
        ds = load_from_disk(osp.join(root, split))
        self.ablation_name = ablation_name
        self.inputs = ds[f"{ablation_name}_input"][:, :input_horizon]
        self.inputs = torch.log(1.0 + torch.from_numpy(self.inputs).float())
        self.orig_labels = ds["citations_label"]
        self.labels = torch.log(1.0 + torch.from_numpy(self.orig_labels).float())
        scaler = TorchStandardScaler()
        self.input_mean, self.input_std = scaler.fit(self.inputs)
        self.inputs = (self.inputs - self.input_mean) / (self.input_std + 1e-6)
        self.labels_mean, self.labels_std = scaler.fit(self.labels)
        self.labels = (self.labels - self.labels_mean) / (self.labels_std + 1e-6)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        orig_label = torch.tensor(self.orig_labels[idx]).float()
        label = self.labels[idx]
        return inputs, label, orig_label


# class GitHubDatasetAblation(Dataset):
#     def __init__(self, root, split, input_horizon, ablation_name=["forks"]):
#         self.root = root
#         self.split = split
#         ds = load_from_disk(osp.join(root, split))
#         self.ablation_name = ablation_name
#         input_arrays = []
#         for name in ablation_name:
#             arr = ds[f"{name}_inputs"][:, :input_horizon]
#             arr = torch.log(1.0 + torch.from_numpy(arr).float())
#             input_arrays.append(arr)

#         self.inputs = torch.cat(input_arrays, dim=-1)

#         self.orig_labels = ds["forks_labels"]
#         self.labels = torch.log(1.0 + torch.from_numpy(self.orig_labels).float())
#         scaler = TorchStandardScaler()
#         self.input_mean, self.input_std = scaler.fit(self.inputs)
#         self.inputs = (self.inputs - self.input_mean) / (self.input_std + 1e-6)
#         self.labels_mean, self.labels_std = scaler.fit(self.labels)
#         self.labels = (self.labels - self.labels_mean) / (self.labels_std + 1e-6)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         inputs = self.inputs[idx]
#         orig_label = torch.tensor(self.orig_labels[idx]).float()
#         label = self.labels[idx]
#         return inputs, label, orig_label


class GitHubDatasetAblation(Dataset):
    def __init__(self, root, split, input_horizon, ablation_name=["forks"]):
        self.root = root
        self.split = split
        ds = load_from_disk(osp.join(root, split))
        self.ablation_name = ablation_name

        # Store individual arrays instead of concatenating
        self.input_arrays = {}
        self.input_stats = {}

        for name in ablation_name:
            arr = ds[f"{name}_inputs"][:, :input_horizon]
            arr = torch.log(1.0 + torch.from_numpy(arr).float())

            # Fit scaler for this array
            scaler = TorchStandardScaler()
            mean, std = scaler.fit(arr)
            normalized_arr = (arr - mean) / (std + 1e-6)

            self.input_arrays[name] = normalized_arr
            self.input_stats[name] = (mean, std)

        self.orig_labels = ds["forks_labels"]
        self.labels = torch.log(1.0 + torch.from_numpy(self.orig_labels).float())
        scaler = TorchStandardScaler()
        self.labels_mean, self.labels_std = scaler.fit(self.labels)
        self.labels = (self.labels - self.labels_mean) / (self.labels_std + 1e-6)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_list = []
        for name in self.ablation_name:
            input_list.append(self.input_arrays[name][idx])

        inputs = torch.cat(input_list, dim=0)
        orig_label = torch.tensor(self.orig_labels[idx]).float()
        label = self.labels[idx]
        return inputs, label, orig_label
