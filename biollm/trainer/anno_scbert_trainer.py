import warnings
import numpy as np
import torch
from .trainer import Trainer
import torch.distributed as dist
from biollm.repo.scbert.utils import get_reduced, dist_cat_tensors
from sklearn.metrics import classification_report, f1_score


class AnnoScbertTrainer(Trainer):
    def __init__(self, args, model, train_loader, val_loader, optimizer, scheduler, criterion, is_master=True):
        super(AnnoScbertTrainer, self).__init__(args, model, train_loader)
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.is_master = is_master

    def train(self, epoch):
        """
        Train the model for one epoch.
        """
        self.model.train()
        losses = 0.0
        sample_sum = 0
        sample_true_sum = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        for batch, batch_data in enumerate(self.train_loader):
            (exp_x, cls_labels) = batch_data
            exp_x = exp_x.to(self.device)
            cls_labels = cls_labels.to(self.device)
            # cls_labels: [batch_size]
            cls_logits = self.model(exp_x)
            loss = self.criterion(cls_logits, cls_labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses += loss.item()
            sample_sum += cls_labels.shape[0]
            sample_true_sum += (cls_logits.argmax(dim=-1) == cls_labels).sum().item()

            if batch % 50 == 0 and batch != 0 and self.is_master:
                self.logger.info('Loss/batch {}: {:.4f}, acc: {:.4f}'.format(batch, loss.item(),
                                                                            sample_true_sum / sample_sum))
        if self.args.distributed:
            sample_sum += get_reduced(sample_sum, self.device, 0, torch.distributed.get_world_size())
            sample_true_sum += get_reduced(sample_true_sum, self.device, 0, torch.distributed.get_world_size())
            losses += get_reduced(losses, self.device, 0, torch.distributed.get_world_size())
        if self.is_master:
            acc = 100 * sample_true_sum / sample_sum
            self.logger.info('Train Epoch {}  > Loss: {:.4f} / Acc: {:.2f}%'
                             .format(epoch, losses / n_batches, acc))
        if self.args.distributed:
            dist.barrier()
        self.scheduler.step()

    def evaluate(self, epoch):
        """
        Train the model for one epoch.
        """
        self.model.eval()
        losses = 0.0
        sample_sum = 0
        sample_true_sum = 0
        predictions = []
        truths = []
        n_batches, n_samples = len(self.val_loader), len(self.val_loader.dataset)
        for batch, batch_data in enumerate(self.val_loader):
            (exp_x, cls_labels) = batch_data
            exp_x = exp_x.to(self.device)
            cls_labels = cls_labels.to(self.device)

            with torch.no_grad():
                cls_logits = self.model(exp_x)
                loss = self.criterion(cls_logits, cls_labels)
                losses += loss.item()
                sample_sum += cls_labels.shape[0]
                sample_true_sum += (cls_logits.argmax(dim=-1) == cls_labels).sum().item()
                predict_labels = cls_logits.argmax(dim=-1)
                predictions.append(predict_labels)
                truths.append(cls_labels)
        if self.args.distributed:
            sample_sum += get_reduced(sample_sum, self.device, 0, torch.distributed.get_world_size())
            sample_true_sum += get_reduced(sample_true_sum, self.device, 0, torch.distributed.get_world_size())
            losses += get_reduced(losses, self.device, 0, torch.distributed.get_world_size())
            predictions = dist_cat_tensors(torch.cat(predictions, dim=0)).detach().cpu().numpy()
            truths = dist_cat_tensors(torch.cat(truths, dim=0)).detach().cpu().numpy()
        else:
            predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
            truths = torch.cat(truths, dim=0).detach().cpu().numpy()
        if self.is_master:
            acc = 100 * sample_true_sum / sample_sum
            f1 = f1_score(truths, predictions, average='macro')
            self.logger.info('Eval Epoch {} > Loss: {:.4f} / Acc: {:.2f}%  f1: {:.2f}%'
                             .format(epoch, losses / n_batches, acc, f1))
        return losses/n_batches

    def predict(self, data_loader):
        """
        Train the model for one epoch.
        """
        self.model.eval()
        predictions = []
        for batch, batch_data in enumerate(data_loader):
            exp_x, _ = batch_data
            exp_x = exp_x.to(self.device)
            with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
                cls_logits = self.model(exp_x)
                predict_labels = cls_logits.argmax(dim=-1)
                predictions.append(predict_labels)
        predictions = torch.cat(predictions, dim=0)
        return predictions.detach().cpu().numpy()

