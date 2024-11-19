#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :integration_task.py
# @Time      :2024/3/6 15:41
# @Author    :Luni Hu

from biollm.biollm.base.bio_task import BioTask
class IntegrationTask(BioTask):
    def __init__(self, config_file):
        super(IntegrationTask, self).__init__(config_file)
        self.gene2ids = self.load_obj.get_gene2idx()
    def train(self):

        pass
    def predict(self):
        pass

