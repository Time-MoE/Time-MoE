#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os

from torch.utils.data import Dataset

from .general_dataset import GeneralDataset
from .binary_dataset import BinaryDataset


class TimeMoEDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.datasets = []

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
                    if GeneralDataset.is_valid_path(fn_path):
                        ds = GeneralDataset(fn_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
                for sub_folder in dirs:
                    folder_path = os.path.join(root, sub_folder)
                    if BinaryDataset.is_valid_path(folder_path):
                        ds = BinaryDataset(folder_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)

        self.num_sequences_list = [len(ds) for ds in self.datasets]
        self.num_sequences = sum(self.num_sequences_list)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        pass
