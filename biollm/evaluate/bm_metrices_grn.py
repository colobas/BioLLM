#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :bm_metrices_grn.py
# @Time      :2024/2/28 10:44
# @Author    :Luni Hu

import networkx as nx
import pickle



def get_genesets_dict(file_path):
    with open(file_path, "rb") as file:
        gene_pw_dict = pickle.load(file)
    return gene_pw_dict


def calculate_jaccard_index(node1, node2, gene_pw_dict):
    set1 = set(gene_pw_dict[node1])
    set2 = set(gene_pw_dict[node2])

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    jaccard_index = intersection / union

    return jaccard_index


def calculate_average_jaccard_index(G, gene_pw_dict):
    G_sub = G.subgraph(list(gene_pw_dict.keys()))

    jaccard_index_list = []
    edge_list = list(G_sub.edges)

    for node1, node2 in edge_list:
        jaccard_index = calculate_jaccard_index(node1, node2, gene_pw_dict)
        jaccard_index_list.append(jaccard_index)

    average_jaccard_index = sum(jaccard_index_list) / G_sub.number_of_edges()

    return average_jaccard_index


def evaluate(G, gene_pw_dict, modularity=True, biological=True):
    modularity_coef = None
    average_jaccard_index = None

    gene_module = nx.community.louvain_communities(G)

    if len(gene_module) > 1:

        if modularity:
            modularity_coef = nx.community.modularity(G, gene_module)

        if biological:
            average_jaccard_index = calculate_average_jaccard_index(G, gene_pw_dict)

        eval_res = {"Modularity": modularity_coef,
                    "Biological Index": average_jaccard_index}

        return eval_res

    else:

        return None
