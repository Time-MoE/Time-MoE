#!/usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np

from time_moe.datasets.ts_dataset import TimeSeriesDataset


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
        for seq_idx, start_idx_in_seq, offset in window_info:
            part_seq = self.dataset[seq_idx][start_idx_in_seq: start_idx_in_seq + offset]
            seq.append(part_seq)
        if len(seq) == 1:
            return seq[0]
        seq = np.concatenate([seq], axis=0)
        return seq


if __name__ == '__main__':
    folder = '/Users/shixiaoming/codes/Time-MoE/tmp_time_moe_dataset/time_300b'
    from time_moe.datasets.time_moe_dataset import TimeMoEDataset
    dataset = TimeMoEDataset(folder, normalization_method=None)

    window_dataset = WindowDataset(dataset, context_length=4096, prediction_length=0)

    print(len(dataset))

    for i in range(len(window_dataset)):
        seq = window_dataset[i]