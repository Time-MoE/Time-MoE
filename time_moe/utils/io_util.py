#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: peter.sxm
@project: TimeMOE
@time: 2024/3/18 21:38
@desc:
"""
import json
import os
import gzip
import pickle

import numpy as np
import yaml


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
        data = load_dill_obj(fn)
    return data


def write_file_by_extension(obj, fn):
    dirname = os.path.dirname(fn)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    if fn.endswith('.json'):
        with open(fn, 'w', encoding='utf-8') as file:
            json.dump(obj, file)
    elif fn.endswith('.jsonl'):
        write_jsonl_to_file(obj, fn)
    elif fn.endswith('.yaml'):
        save_yaml_file(obj, fn)
    elif fn.endswith('.npy'):
        np.save(fn, obj)
    elif fn.endswith('.npy.gz'):
        with gzip.GzipFile(fn, 'w') as file:
            np.save(file, obj, allow_pickle=True)
    else:
        # elif fn.endswith('.pkl') or fn.endswith('.pickle'):
        dump_dill_obj(obj, fn)


def write_jsonl_to_file(obj_list, jsonl_fn):
    with open(jsonl_fn, 'w', encoding='utf-8') as file:
        for obj in obj_list:
            file.write(json.dumps(obj, ensure_ascii=False) + '\n')


def read_jsonl_to_list(jsonl_fn):
    with open(jsonl_fn, 'r', encoding='utf-8') as file:
         out = []
         for line in file.readlines():
             if line == '':
                 continue
             obj = json.loads(line)
             out.append(obj)
    return out


def load_yaml_file(fn):
    if isinstance(fn, str):
        with open(fn, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config
    else:
        return fn


def save_yaml_file(obj, fn):
    folder = os.path.dirname(fn)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(fn, 'w', encoding="utf-8") as file:
        yaml.dump(obj, file)
    return fn


def dump_dill_obj(obj, fn):
    import dill
    with open(fn, 'wb') as file:
        dill.dump(obj, file)


def load_dill_obj(fn):
    import dill
    with open(fn, 'rb') as file:
        return dill.load(file)


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
