#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: utils.py.py
@time: 2024/3/3 11:26
"""
import munch
import toml
import json
import numpy as np
import torch
from functools import wraps
import pynvml


def load_config(config_file):
    args = munch.munchify(toml.load(config_file))
    if args.model_used in ('scgpt', 'scmamba'):
        with open(args.model_param_file, 'r') as fd:
            params = json.load(fd)
        for p in params:
            if p not in args:
                args[p] = params[p]
    return args


def gene2vec_embedding(g2v_file, g2v_genes):
    gene2vec_weight = np.load(g2v_file)
    gene_emb_dict = {}
    with open(g2v_genes, 'r') as fd:
        gene_list = [line.strip('\n') for line in fd]
    for i in range(len(gene_list)):
        gene_emb_dict[gene_list[i]] = gene2vec_weight[i]
    return gene_emb_dict


def cal_model_params(model: torch.nn.Module) -> int:
    """
    calculate model parameters
    """
    model_param_count = 0
    for param in model.parameters():
        model_param_count += param.numel()
    return model_param_count


def cal_gpu_memory(gpu_index: int):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return memory_info.used / (1024**3)


def gpu_memory(gpu_index: int):
    def gpu_resource(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            memory = cal_gpu_memory(gpu_index)
            print('gpu: ', memory, 'G')
        return wrapper
    return gpu_resource


def get_reduced(tensor, device, dest_device):
    """
    将不同GPU上的变量或tensor集中在主GPU上，并得到均值
    """
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / torch.distributed.get_world_size()
    return tensor_mean


def distributed_concat(tensor, num_total_examples, world_size):
    """
    合并不同进程的inference结果
    """
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]