#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import torch.distributed as dist


def is_master_process():
    rank = os.getenv('RANK')
    if (rank is None or rank == '0') and is_local_rank_0():
        return True
    else:
        return False


def is_local_rank_0():
    local_rank = os.getenv('LOCAL_RANK')
    if local_rank is None or local_rank == '0':
        return True
    else:
        return False


def get_local_world_size():
    import torch
    local_world_size = os.getenv('LOCAL_WORLD_SIZE')
    if local_world_size is None:
        num_gpus = torch.cuda.device_count()
        local_world_size = num_gpus or 1
    else:
        local_world_size = int(local_world_size)
    return local_world_size


def get_world_size():
    try:
        world_size = dist.get_world_size()
        return world_size
    except Exception:
        pass
    world_size = os.getenv('WORLD_SIZE')
    if world_size is None:
        world_size = 1
    else:
        world_size = int(world_size)
    return world_size
