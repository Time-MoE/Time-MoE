#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: peter.sxm
@project: TimeMOE
@time: 2024/8/4 11:18
@desc:
"""
import argparse
import os
import random
import math
import numpy as np

from time_moe.utils.io_util import read_file_by_extension, write_file_by_extension


def save_array_to_bin(arr, fn):
    with open(fn, mode='wb') as file:
        arr.tofile(file)


def process_to_bin(data_path, out_folder, shuffle: bool = True, dtype: str = 'float32'):
    try:
        max_chunk_size = (1 << 30) * 5
        data = read_file_by_extension(data_path)
        if len(data) == 0:
            print('Sequence is empty:', data_path)
            return 0

        if shuffle:
            random.shuffle(data)

        meta = {}
        meta['num_sequences'] = len(data)
        meta['dtype'] = dtype
        meta['files'] = {}
        meta['scales'] = []

        idx = 0
        file_name_format = 'data-{}-of-{}.bin'
        sequence = []
        for d in data:
            seq = d['sequence']

            if 'meta' in d and 'mean' in d['meta']:
                meta['scales'].append(
                    {
                        'offset': idx,
                        'mean': d['meta']['mean'],
                        'std': d['meta']['std'],
                        'length': len(seq),
                        'id': d.get('id')
                    }
                )
            else:
                meta['scales'].append(
                    {
                        'offset': idx,
                        'length': len(seq),
                        'id': d.get('id')
                    }
                )

            idx += len(seq)
            sequence.append(seq)

        sequence = np.concatenate(sequence, axis=0, dtype=dtype)

        # chunk
        memory_size = sequence.nbytes
        num_chunks = math.ceil(memory_size / max_chunk_size)
        chunk_length = math.ceil(len(sequence) / num_chunks)

        os.makedirs(out_folder, exist_ok=True)
        for i in range(0, num_chunks):
            start_idx = i * chunk_length
            end_idx = start_idx + chunk_length
            sub_seq = sequence[start_idx: end_idx]
            sub_fn = file_name_format.format(i + 1, num_chunks)

            out_fn = os.path.join(out_folder, sub_fn)
            save_array_to_bin(sub_seq, out_fn)

            meta['files'][sub_fn] = len(sub_seq)

        # save meta
        write_file_by_extension(meta, os.path.join(out_folder, 'meta.json'))
        return len(sequence)
    except Exception as e:
        print(f'>> process fail in file {data_path}: ', str(e))
        return 0


def process_src_folder_to_tgt_folder(src_data_folder, out_folder):
    if os.path.isdir(src_data_folder):
        for root, dir, files in os.walk(src_data_folder):
            for file in files:
                if (file.endswith('.jsonl')
                        or file.endswith('.npy')
                        or file.endswith('.npy.gz')
                        or file.endswith('.pkl')
                ):
                    rel_folder = os.path.relpath(root, src_data_folder)
                    data_path = os.path.join(root, file)

                    file_out_folder = os.path.join(out_folder, rel_folder, file.split('.')[0])
                    print('processing', file)
                    num_tokens = process_to_bin(
                        data_path=data_path,
                        out_folder=file_out_folder,
                        shuffle=False,
                        dtype='float32'
                    )
                    print('-- process ', file, 'to', file_out_folder, f'{num_tokens / 1e6:.3f} M tokens')
    else:
        process_to_bin(
            data_path=src_data_folder,
            out_folder=out_folder,
            shuffle=False,
            dtype='float32'
        )
        print('-- process ', src_data_folder, 'to', out_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i',
        help='Input file or folder.'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output folder.'
    )
    args = parser.parse_args()

    process_src_folder_to_tgt_folder(
        args.input,
        args.output,
    )