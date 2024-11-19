#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :bm_metrices_anno
# @Time      :2024/3/5 16:20
# @Author    :Luni Hu

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(labels, predictions):

    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')

    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }