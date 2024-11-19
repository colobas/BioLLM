#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: network_task.py
@time: 2024/3/3 15:15
"""

import networkx as nx
import torch
from biollm.base.bio_task import BioTask
import numpy as np
import os
from biollm.model.grn import GeneEmbedding
import matplotlib.pyplot as plt


class GrnTask(BioTask):
    def __init__(self, cfs_file):
        super(GrnTask, self).__init__(cfs_file)
        self.gene2ids = self.load_obj.get_gene2idx()

    def grn_analysis(self, quantile_cutoff, finetune=False):
        if self.args.quantile_cutoff > 0:
            threshold = self.args.quantile_cutoff
        elif quantile_cutoff is not None:
            threshold = quantile_cutoff
        else:
            threshold = 0.8
        if not finetune:
            genes_list = list(self.gene2ids.keys())
            gene_ids = np.array(list(self.gene2ids.values()))
            gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(self.device)
            ids_embedding = self.load_obj.get_gene_embedding(gene_ids)
            genes_embedding = {genes_list[i]: ids_embedding[i] for i in range(len(genes_list))}
        else:
            adata = self.read_h5ad()
            genes_list = [gene for gene in adata.var.index.to_list() if gene in self.gene2ids]
            gene_ids = np.array([self.gene2ids[gene] for gene in genes_list])
            gene_ids = torch.tensor(gene_ids, dtype=torch.long).to(self.device)
            ids_embedding = self.load_obj.get_gene_embedding(gene_ids)
            genes_embedding = {genes_list[i]: ids_embedding[i] for i in range(len(genes_list))}
        self.logger.info('gene gene embedding for {} genes.'.format(len(genes_embedding)))
        grn_embed = GeneEmbedding(genes_embedding)
        g = grn_embed.generate_network(threshold=threshold)
        #nx.write_adjlist(g, os.path.join(self.args.output_dir, '{}_grn_network_adjlist.gz'.format(self.args.model_type)))
        return g

    @staticmethod
    def plot_network(G, output_path=None, thresh=0.4):
        # Plot the cosine similarity network; strong edges (> select threshold) are highlighted
        plt.figure(figsize=(20, 20))
        widths = nx.get_edge_attributes(G, 'weight')
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > thresh]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= thresh]

        pos = nx.spring_layout(G, k=0.4, iterations=15, seed=3)
        width_large = {}
        width_small = {}
        for i, v in enumerate(list(widths.values())):
            if v > thresh:
                width_large[list(widths.keys())[i]] = v * 10
            else:
                width_small[list(widths.keys())[i]] = max(v, 0) * 10

        nx.draw_networkx_edges(G, pos,
                               edgelist=width_small.keys(),
                               width=list(width_small.values()),
                               edge_color='lightblue',
                               alpha=0.8)
        nx.draw_networkx_edges(G, pos,
                               edgelist=width_large.keys(),
                               width=list(width_large.values()),
                               alpha=0.5,
                               edge_color="blue",
                               )
        # node labels
        nx.draw_networkx_labels(G, pos, font_size=25, font_family="sans-serif")
        # edge weight labels
        d = nx.get_edge_attributes(G, "weight")
        edge_labels = {k: d[k] for k in elarge}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.show()
        if output_path is not None:
            plt.savefig(output_path)