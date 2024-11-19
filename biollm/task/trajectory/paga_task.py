#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 15:32
# @Author  : qiuping
# @File    : paga_task.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/7/1 15:32  create file. 
"""
import scanpy as sc
from biollm.base.bio_task import BioTask


class PagaTask(BioTask):
    def __init__(self, cfs_file):
        super().__init__(cfs_file)

    def get_cell_emb(self):
        pass

    def paga(self, adata):
        sc.pp.neighbors(adata, use_rep=self.args['use_model'] + '_emb')
        sc.tl.draw_graph(adata)
        return adata
