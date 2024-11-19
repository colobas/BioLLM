#!/usr/bin/env python3
# coding: utf-8
"""
@file: trainer.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/03  create file.
"""
import torch
from biollm.utils.log_manager import LogManager


class Trainer(object):
    def __init__(self, args, model, train_loader):
        self.args = args
        self.logger = LogManager().logger
        self.device = torch.device(self.args.device)
        self.model = model
        self.train_loader = train_loader

    def train(self, *args, **kwargs):
        raise NotImplementedError('Please implement!')

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError('Please implement!')

    def predict(self, *args, **kwargs):
        raise NotImplementedError('Please implement!')
