#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import numpy as np

from .ts_dataset import TimeSeriesDataset
from .general_dataset import GeneralDataset
from .binary_dataset import BinaryDataset


class TimeMoEDataset(TimeSeriesDataset):
    def __init__(self, data_folder, normalization_method=None):
        self.data_folder = data_folder
        self.normalization_method = normalization_method
        self.datasets = []
        if normalization_method is None:
            self.normalization_method = None
        elif isinstance(normalization_method, str):
            if normalization_method.lower() == 'max':
                self.normalization_method = max_scaler
            elif normalization_method.lower() == 'zero':
                self.normalization_method = zero_scaler
            else:
                raise ValueError(f'Unknown normalization method: {normalization_method}')
        else:
            self.normalization_method = normalization_method

        if BinaryDataset.is_valid_path(self.data_folder):
            ds = BinaryDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        elif GeneralDataset.is_valid_path(self.data_folder):
            ds = GeneralDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        else:
            # walk through the data_folder
            for root, dirs, files in os.walk(self.data_folder):
                for file in files:
                    fn_path = os.path.join(root, file)
                    if file != BinaryDataset.meta_file_name and GeneralDataset.is_valid_path(fn_path):
                        ds = GeneralDataset(fn_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
                for sub_folder in dirs:
                    folder_path = os.path.join(root, sub_folder)
                    if BinaryDataset.is_valid_path(folder_path):
                        ds = BinaryDataset(folder_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)

        self.cumsum_lengths = [0]
        for ds in self.datasets:
            self.cumsum_lengths.append(
                self.cumsum_lengths[-1] + len(ds)
            )
        self.num_sequences = self.cumsum_lengths[-1]

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}')
        elif seq_idx < 0:
            raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        seq = self.datasets[dataset_idx][dataset_offset]

        if self.normalization_method is not None:
            seq = self.normalization_method(seq)
        return seq

    def get_sequence_length_by_idx(self, seq_idx):
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}')
        elif seq_idx < 0:
            raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_sequence_length_by_idx(dataset_offset)


class WindowDataset:
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length

        self.window_info_list = []
        n_seqs = len(self.dataset)

        cur_window_info = []
        num_cur_remaining_points = self.window_size
        for seq_idx in range(n_seqs):
            seq_len = self.dataset.get_sequence_length_by_idx(seq_idx)
            remaining_seq_len = seq_len
            while remaining_seq_len > 0:
                if remaining_seq_len < num_cur_remaining_points:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, remaining_seq_len)
                    )

                    # update states
                    num_cur_remaining_points -= remaining_seq_len
                    remaining_seq_len = 0
                else:
                    # add the part of this seq to cur_window
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, num_cur_remaining_points)
                    )

                    # update states
                    remaining_seq_len -= num_cur_remaining_points
                    self.window_info_list.append(cur_window_info)

                    # reset current window
                    num_cur_remaining_points = self.window_size
                    cur_window_info = []

        if num_cur_remaining_points > 0:
            # TODO drop last temporarily
            pass

    def __len__(self):
        return len(self.window_info_list)

    def __getitem__(self, window_idx):
        window_info = self.window_info_list[window_idx]
        seq = []
        for seq_idx, start_idx_in_seq, offset  in window_info:
            part_seq = self.dataset[seq_idx][start_idx_in_seq: start_idx_in_seq + offset]
            seq.append(part_seq)
        seq = np.concatenate([seq], axis=0)
        return seq



def zero_scaler(seq):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    std_val = seq.std()
    if std_val == 0:
        normed_seq = seq
    else:
        normed_seq = (seq - seq.mean()) / std_val

    return normed_seq


def max_scaler(seq):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    max_val = np.abs(seq).max()
    if max_val == 0:
        normed_seq = seq
    else:
        normed_seq = seq / max_val

    return normed_seq


def binary_search(sorted_list, value):
    low = 0
    high = len(sorted_list) - 1
    best_index = -1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] <= value:
            best_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_index
