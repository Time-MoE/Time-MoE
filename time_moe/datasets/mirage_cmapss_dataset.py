# time_moe/datasets/mirage_cmapss_dataset.py
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from datasets import load_dataset


@dataclass
class MirageWindowItem:
    input_ids: np.ndarray   # [L, F]
    labels: np.ndarray      # [L, F]
    loss_masks: np.ndarray  # [L]
    entity_id: int
    timestamp: float


class MirageCmapssWindowDataset(Dataset):
    """
    C-MAPSS (HF: nominal-io/nasa-turbofan-degradation) -> snapshot windows for MIRAGE.
    Each item is a fixed-length window with:
      - input_ids:  [L, F]
      - labels:     [L, F] (next-step shift)
      - loss_masks: [L]    (all ones, no padding in this dataset)
      - entity_id:  engine id
      - timestamp:  cycle at the last input step (weak order cue)
    """
    def __init__(
        self,
        hf_name: str = "nominal-io/nasa-turbofan-degradation",
        split: str = "train",
        context_length: int = 64,
        stride: int = 16,
        normalize: bool = True,
        seed: int = 9899,
    ):
        super().__init__()
        self.hf_name = hf_name
        self.split = split
        self.context_length = int(context_length)
        self.stride = int(stride)
        self.normalize = bool(normalize)
        self.seed = int(seed)

        ds = load_dataset(self.hf_name, split=self.split)
        df = ds.to_pandas()

        self.id_col = "engine"
        self.time_col = "cycle"

        drop_cols = [c for c in ["Unnamed: 0", self.id_col, self.time_col] if c in df.columns]
        self.feature_cols = [c for c in df.columns if c not in drop_cols]

        # sort
        df = df.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

        # build global normalization stats (train-split stats)
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        if self.normalize:
            self.mean = X.mean(axis=0, keepdims=True)
            self.std = X.std(axis=0, keepdims=True)
            self.std = np.where(self.std < 1e-6, 1.0, self.std)
        else:
            self.mean = np.zeros((1, X.shape[1]), dtype=np.float32)
            self.std = np.ones((1, X.shape[1]), dtype=np.float32)

        # group by engine
        engine_ids = sorted(df[self.id_col].unique().tolist())
        self.engine_ids = engine_ids

        win_len_plus_one = self.context_length + 1  # for shift labels
        items: List[MirageWindowItem] = []
        entity_to_indices: Dict[int, List[int]] = {}

        for eid in engine_ids:
            sub = df[df[self.id_col] == eid].sort_values(self.time_col)
            x_tf = sub[self.feature_cols].to_numpy(dtype=np.float32)
            t_arr = sub[self.time_col].to_numpy(dtype=np.float32)

            # normalize
            x_tf = (x_tf - self.mean) / self.std

            T = x_tf.shape[0]
            if T < win_len_plus_one:
                continue

            # sliding windows (no padding for CMAPSS)
            # start in [0, T - (L+1)] with step=stride
            for start in range(0, T - win_len_plus_one + 1, self.stride):
                end = start + win_len_plus_one
                seq = x_tf[start:end]  # [L+1, F]

                input_ids = seq[:-1]   # [L, F]
                labels = seq[1:]       # [L, F]
                loss_masks = np.ones((self.context_length,), dtype=np.float32)

                # timestamp = cycle at the last input step (end-2 in the raw seq)
                ts = float(t_arr[end - 2])

                idx = len(items)
                items.append(
                    MirageWindowItem(
                        input_ids=input_ids.astype(np.float32),
                        labels=labels.astype(np.float32),
                        loss_masks=loss_masks,
                        entity_id=int(eid),
                        timestamp=ts,
                    )
                )
                entity_to_indices.setdefault(int(eid), []).append(idx)

        self.items = items
        self.entity_to_indices = entity_to_indices

        # deterministic behavior (sampler will also seed itself if needed)
        random.seed(self.seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        return {
            "input_ids": it.input_ids,     # np.float32 [L, F]
            "labels": it.labels,           # np.float32 [L, F]
            "loss_masks": it.loss_masks,   # np.float32 [L]
            "entity_ids": np.int64(it.entity_id),
            "timestamps": np.float32(it.timestamp),
        }


class EntityBatchSampler(Sampler[List[int]]):
    """
    Yield batches shaped as: B entities * K windows per entity -> total batch size B*K.
    This enforces MIRAGE's within-entity ranking structure.

    Args:
        entity_to_indices: dict[entity_id] -> list[window_idx]
        batch_entities: B
        windows_per_entity: K
        num_batches: number of batches per epoch/iteration loop
    """
    def __init__(
        self,
        entity_to_indices: Dict[int, List[int]],
        batch_entities: int,
        windows_per_entity: int,
        num_batches: int,
        seed: int = 9899,
        shuffle_entities: bool = True,
        replace_entities: bool = False,
        replace_windows: bool = False,
    ):
        self.entity_to_indices = entity_to_indices
        self.entities = sorted(list(entity_to_indices.keys()))
        self.B = int(batch_entities)
        self.K = int(windows_per_entity)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.shuffle_entities = bool(shuffle_entities)
        self.replace_entities = bool(replace_entities)
        self.replace_windows = bool(replace_windows)

        if len(self.entities) == 0:
            raise ValueError("EntityBatchSampler: entity_to_indices is empty.")

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        entities = self.entities

        for _ in range(self.num_batches):
            # sample B entities
            if self.replace_entities or self.B > len(entities):
                chosen_entities = [rng.choice(entities) for _ in range(self.B)]
            else:
                if self.shuffle_entities:
                    chosen_entities = rng.sample(entities, self.B)
                else:
                    # deterministic: take first B (useful for debugging)
                    chosen_entities = entities[: self.B]

            batch_indices: List[int] = []
            for eid in chosen_entities:
                idxs = self.entity_to_indices[eid]
                if len(idxs) == 0:
                    continue
                # sample K windows within entity
                if self.replace_windows or self.K > len(idxs):
                    picked = [rng.choice(idxs) for _ in range(self.K)]
                else:
                    picked = rng.sample(idxs, self.K)
                batch_indices.extend(picked)

            # ensure exact length B*K (if some entity had no windows, pad by resampling globally)
            while len(batch_indices) < self.B * self.K:
                eid = rng.choice(entities)
                idxs = self.entity_to_indices[eid]
                batch_indices.append(rng.choice(idxs))

            yield batch_indices[: self.B * self.K]


def mirage_collate_fn(batch: List[dict]) -> dict:
    """
    Collate list of items into tensors.
    Returns:
      input_ids:  [N, L, F]
      labels:     [N, L, F]
      loss_masks: [N, L]
      entity_ids: [N]
      timestamps: [N]
    """
    input_ids = torch.from_numpy(np.stack([b["input_ids"] for b in batch], axis=0)).float()
    labels = torch.from_numpy(np.stack([b["labels"] for b in batch], axis=0)).float()
    loss_masks = torch.from_numpy(np.stack([b["loss_masks"] for b in batch], axis=0)).float()
    entity_ids = torch.from_numpy(np.stack([b["entity_ids"] for b in batch], axis=0)).long()
    timestamps = torch.from_numpy(np.stack([b["timestamps"] for b in batch], axis=0)).float()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_masks,
        "entity_ids": entity_ids,
        "timestamps": timestamps,
    }
