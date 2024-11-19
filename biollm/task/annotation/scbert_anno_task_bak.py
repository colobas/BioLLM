#!/usr/bin/env python3
# coding: utf-8
"""
@file: pert_task.py
@description:
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/02  create file.
"""
import pickle

from biollm.base.bio_task import BioTask
from biollm.model.loss import FocalLoss
import os
from torch import nn
import torch
from biollm.trainer.anno_scbert_trainer import AnnoScbertTrainer
import time
import copy
import scanpy as sc
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from biollm.model.annotation import ScbertClassification
from biollm.repo.st_performer.model.learn_rate import CosineAnnealingWarmupRestarts


class ScbertAnnoTask(BioTask):
    def __init__(self, config_file):
        super(ScbertAnnoTask, self).__init__(config_file)
        # init the func for the trainer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, eps=1e-4)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=self.args.lr,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )
        self.criterion = nn.CrossEntropyLoss(weight=None)
        # set the distributed trainning
        self.is_master = int(os.environ['RANK']) == 0 if self.args.distributed else True
        if self.is_master:
            self.logger.info(self.args)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.distributed:
            local_rank = int(os.environ['LOCAL_RANK'])
            self.logger.info(f'local rank: {local_rank}')
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
        # make the data loader for the trainer
        split_data = True if self.args.finetune else False
        self.train_loader, self.val_loader = self.make_dataloader(split_data=split_data)
        # for the finetune setting
        if self.args.finetune:
            self.load_obj.freezon_model(keep_layers=[-2])
            self.model.to_out = ScbertClassification(h_dim=128,
                                                     class_num=len(self.load_obj.label_dict),
                                                     max_seq_len=self.args.max_seq_len, dropout=0.).to(
                self.args.device)
        if self.args.distributed:
            self.args.device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
            self.args.local_rank = local_rank
            self.criterion = self.criterion.to(local_rank)
            self.model = self.model.to(self.args.device)
            self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                                 output_device=self.args.local_rank, find_unused_parameters=False)
        else:
            self.model = self.model.to(self.args.device)

    def make_dataloader(self, split_data=True):
        adata = sc.read_h5ad(self.args.input_file)
        train_dataset, val_dataset = self.load_obj.load_dataset(adata, self.args.do_preprocess, split_data=split_data,
                                                                label_key=self.args.label_key)
        random_sample = True if self.args.finetune else False
        train_loader = self.load_obj.get_dataloader(train_dataset, random_sample=random_sample)
        val_loader = self.load_obj.get_dataloader(val_dataset,
                                                  random_sample=random_sample) if val_dataset is not None else None
        return train_loader, val_loader

    def run(self):
        best_val_loss = float("inf")
        best_model = None
        patience = 0

        trainer = AnnoScbertTrainer(
            args=self.args,
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            is_master=self.is_master,
            scheduler=self.scheduler
        )
        if self.args.finetune:
            epoch_start_time = time.time()
            if self.args.distributed:
                dist.barrier()
            for epoch in range(1, self.args.epochs + 1):
                trainer.train(epoch)
                val_loss = trainer.evaluate(epoch)
                elapsed = time.time() - epoch_start_time
                if self.is_master:
                    self.logger.info(
                        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                        f"valid loss: {val_loss:5.4f} |"
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(self.model)
                    if self.is_master:
                        self.logger.info(f"Best model with score {best_val_loss:5.4f}")
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.args.early_stop:
                        if self.is_master:
                            self.logger.info(f"Early stop at epoch {epoch}")
                        break

                torch.save(
                    self.model.state_dict(),
                    f"{self.args.output_dir}/model_{epoch}.pt",
                )

            if best_model is not None:
                torch.save(best_model.state_dict(), self.args.output_dir + "/best_model.pt")
            self.model = best_model
        if self.args.predict:
            pred = trainer.predict(self.train_loader)
            adata = sc.read_h5ad(self.args.input_file)
            adata.obs['pred_celltype'] = pred
            adata.write_h5ad(os.path.join(self.output_dir, 'scbert_pred.h5ad'))


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1]
    # config_file = '../../configs/anno/scbert_anno.toml'
    obj = ScbertAnnoTask(config_file)
    obj.run()
