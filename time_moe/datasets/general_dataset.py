#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import yaml
import gzip
import pickle
import numpy as np

from .ts_dataset import TimeSeriesDataset


class GeneralDataset(TimeSeriesDataset):
    def __init__(self, data_path):
        self.data = read_file_by_extension(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        return self.data[seq_idx]

    def get_sequence_length_by_idx(self, seq_idx):
        return len(self.data[seq_idx])


def read_file_by_extension(fn):
    if fn.endswith('.json'):
        with open(fn, encoding='utf-8') as file:
            data = json.load(file)
    elif fn.endswith('.jsonl'):
        data = read_jsonl_to_list(fn)
    elif fn.endswith('.yaml'):
        data = load_yaml_file(fn)
    elif fn.endswith('.npy'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npz'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npy.gz'):
        with gzip.GzipFile(fn, 'r') as file:
            data = np.load(file, allow_pickle=True)
    elif fn.endswith('.pkl') or fn.endswith('.pickle'):
        data = load_pkl_obj(fn)
    else:
        raise RuntimeError(f'Unknown file extension: {fn}')
    return data


def read_jsonl_to_list(jsonl_fn):
    with open(jsonl_fn, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]


def load_yaml_file(fn):
    if isinstance(fn, str):
        with open(fn, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config
    else:
        return fn


def load_pkl_obj(fn):
    out_list = []
    with open(fn, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                out_list.append(data)
            except EOFError:
                break
    if len(out_list) == 0:
        return None
    elif len(out_list) == 1:
        return out_list[0]
    else:
        return out_list
