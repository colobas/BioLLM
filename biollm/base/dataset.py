#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: dataset.py
@time: 2024/3/5 19:46
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass
from anndata import AnnData
from torch.utils.data import Dataset


@dataclass
class DataPrepare:
    adata: AnnData
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    do_mask: bool = False
    do_sort: bool = False
    mask_probability: float = 0.15
    mask_token_id: Optional[int] = None
    mask_value: int = -1
    ignore_mask_token = []
    ignore_mask_value = []
    max_length: Optional[int] = None
    sampling: bool = True
    keep_first_n_tokens: int = 1

    def check_params(self):
        if self.pad_token_id is None:
            raise ValueError("`pad_token_id` is required if `do_padding`.")
        if self.max_length is None:
            raise ValueError("`max_length` is required if `do_padding`.")

        if self.mask_probability <= 0 or self.mask_probability >= 1:
            raise ValueError("`mlm_probability` must be between 0 and 1.")

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

    def anndata2dict(self, include_zero_gene=True, cls_id=None, cls_index=0):
        result = []
        for i in range(len(self.adata.obs)):
            gene_ids = self.adata.var['gene_ids']
            values = self.adata.obs['express_x']
            if not include_zero_gene:
                idx = np.nonzero(values)[0]
                values = values[idx]
                gene_ids = gene_ids[idx]
            max_len = self.max_length - 1 if cls_id is not None else self.max_length
            if len(gene_ids) > max_len:
                use_idx = np.random.choice(len(gene_ids), max_len, replace=False)
                gene_ids = gene_ids[use_idx]
                values = gene_ids[use_idx]
            if len(gene_ids) < max_len:
                gene_ids = np.hstack(
                    (gene_ids, np.full((max_len - len(gene_ids)), self.pad_token_id, dtype=gene_ids.dtype)))
                values = np.hstack((values, np.full((max_len - len(gene_ids)), self.pad_value, dtype=values.dtype)))
            if cls_id is not None:
                gene_ids = np.insert(gene_ids, cls_index, cls_id)
                values = np.insert(values, cls_index, 0)
            row = {
                "gene_ids": gene_ids,
                "express_x": values,
                "celltype": self.adata.obs['celltype'][i],
                "batch": self.adata.obs['batch'][i]
            }
            if self.do_mask:
                # mask_gene_ids, _ = self.mask(gene_ids, ignore_mask_tokens=self.ignore_mask_token,
                #                              mask_token_id=self.mask_token_id, pad_token_id=self.pad_token_id)
                mask_express_x, _ = self.mask(values, ignore_mask_tokens=self.ignore_mask_value,
                                              mask_token_id=self.mask_value, pad_token_id=self.pad_value)
                row['mask_express_x'] = mask_express_x
            if self.do_sort:
                pass
            result.append(row)
        return result

    def mask(self, x, ignore_mask_tokens, mask_token_id, pad_token_id, mask_prob=0.15, keep_mask_pro=0.8,
             random_replace_mask_pro=0.1, token_nums=None):
        mask_tokens = self.is_mask_tokens(x, ignore_mask_tokens)
        mask_index = np.nonzero(mask_tokens)[0]
        mask_num = np.int32(np.ceil(len(mask_index) * mask_prob * keep_mask_pro))
        keep_mask_index = np.random.choice(mask_index, mask_num, replace=False)
        mask_x = np.copy(x)
        mask_x[keep_mask_index] = mask_token_id
        labels = np.full_like(x, pad_token_id)
        labels[keep_mask_index] = x[keep_mask_index]
        if random_replace_mask_pro:
            assert token_nums is not None, "error: token_nums must be set if random_replace_mask_pro > 0."
            token_ids = np.arange(token_nums)
            replace_tokens = mask_tokens
            replace_tokens[keep_mask_index] = False
            random_replace_num = np.ceil(len(mask_index) * mask_prob * random_replace_mask_pro).astype(np.int32)
            random_replace_index = np.random.choice(np.nonzero(replace_tokens)[0], random_replace_num,
                                                    replace=False)
            labels[random_replace_index] = x[random_replace_index]
            random_token_ids = np.random.choice(token_ids, random_replace_num, replace=True)
            mask_x[random_replace_index] = random_token_ids
        return mask_x, labels

    @staticmethod
    def is_mask_tokens(x, ignore_mask_tokens):
        ignore_mask_tokens = np.array(ignore_mask_tokens)
        no_mask = np.isin(x, ignore_mask_tokens)
        return ~no_mask


class ScDataset(Dataset):

    def __init__(self, data_list):
        super().__init__()
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]


