# -*- coding: utf-8 -*-

import argparse

import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from biollm.repo.scbert.performer_pytorch.performer_pytorch import PerformerLM
import scanpy as sc
from biollm.repo.scbert.utils import *
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')

args = parser.parse_args()

from biollm.dataset.scbert_dataset import SCDataset, make_scbert_adata
from biollm.task.annotation.anno_task_scbert import ScbertAnnoTask
from biollm.model.annotation import ScbertClassification
from biollm.base.load_scbert import LoadScbert
from biollm.evaluate.bm_metrices_anno import compute_metrics
from scipy.sparse import issparse


def make_dataloader(data_path):
    adata = sc.read_h5ad(data_path)
    data = adata.X.toarray() if issparse(adata.X) else adata.X

    label_dict, label = np.unique(np.array(adata.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    # store the label dict and label for prediction
    with open('label_dict', 'wb') as fp:
        pkl.dump(label_dict, fp)
    with open('label', 'wb') as fp:
        pkl.dump(label, fp)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    for index_train, index_val in sss.split(data, label):
        data_train, label_train = data[index_train], label[index_train]
        # data_tmp, label_tmp = data[index_val], label[index_val]
        data_val, label_val = data[index_val], label[index_val]
        train_dataset = SCDataset(data_train, 7, label_train)
        val_dataset = SCDataset(data_val, 7, label_val)
    #
    # sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.333, random_state=SEED)
    # for index_test, index_val in sss1.split(data_tmp, label_tmp):
    #     data_test, label_test = data[index_test], label[index_test]
    #     data_val, label_val = data[index_val], label[index_val]
    #     test_dataset = SCDataset(data_test, 7, label_test)
    #     val_dataset = SCDataset(data_val, 7, label_val)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    # test_sampler = DistributedSampler(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
    return train_loader, val_loader, label_dict


def load_model(label_dict):
    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=POS_EMBED_USING,
        g2v_file='/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/train_data/gene2vec/gene2vec_16906.npy'
    )
    path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/scBERT/ckpt/panglao_pretrain.pth'
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    model_param_count = 0
    ft_param_count = 0
    for param in model.parameters():
        model_param_count += param.numel()
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = True
        ft_param_count += param.numel()
    for param in model.performer.net.layers[-2].parameters():
        param.requires_grad = True
        ft_param_count += param.numel()
    print(f"Total pretrain-model Encoder Params {model_param_count}")
    print(f"The pretrain_model Encoder Params for training in finetune after freezon: {ft_param_count}")
    model.to_out = ScbertClassification(max_seq_len=SEQ_LEN, dropout=0., h_dim=128,
                                        class_num=len(label_dict))
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model


def train(model, train_loader, val_loader):
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )
    loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
    dist.barrier()
    min_loss = 100000
    for i in range(1, EPOCHS+1):
        train_loader.sampler.set_epoch(i)
        model.train()
        dist.barrier()
        running_loss = 0.0
        cum_acc = 0.0
        for index, (data, labels) in enumerate(train_loader):
            index += 1
            data, labels = data.to(device), labels.to(device)
            if index % GRADIENT_ACCUMULATION != 0:
                with model.no_sync():
                    logits = model(data)
                    loss = loss_fn(logits, labels)
                    loss.backward()
            if index % GRADIENT_ACCUMULATION == 0:
                logits = model(data)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final = softmax(logits)
            final = final.argmax(dim=-1)
            pred_num = labels.size(0)
            correct_num = torch.eq(final, labels).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
        if is_master:
            print(f'==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        dist.barrier()
        scheduler.step()
        val_loss = evaluate(i, model, val_loader, loss_fn)
        # evaluate(i, model, test_loader, loss_fn, data_type='Test')
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), f"{ckpt_dir}/model_best.pt")
            # save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
        #     evaluate(i, model, test_loader, loss_fn, data_type='Test')


def evaluate(epoch, model, val_loader, loss_fn, data_type='Validation'):
    model.eval()
    dist.barrier()
    running_loss = 0.0
    predictions = []
    truths = []
    with torch.no_grad():
        for index, (data_v, labels_v) in enumerate(val_loader):
            index += 1
            data_v, labels_v = data_v.to(device), labels_v.to(device)
            logits = model(data_v)
            loss = loss_fn(logits, labels_v)
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
            predictions.append(final)
            truths.append(labels_v)
        del data_v, labels_v, logits, final_prob, final
        # gather
        predictions = dist_cat_tensors(torch.cat(predictions, dim=0))
        # predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
        truths = dist_cat_tensors(torch.cat(truths, dim=0))
        # truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
        no_drop = predictions != -1
        predictions = np.array((predictions[no_drop]).cpu())
        truths = np.array((truths[no_drop]).cpu())
        cur_acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions, average='macro')
        val_loss = running_loss / index
        val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            print(f'==  Epoch: {epoch} | {data_type} Loss: {val_loss:.6f} | Acc: {cur_acc:.6f} | F1 Score: {f1:.6f}  ==')
    del predictions, truths
    return val_loss


if __name__ == '__main__':
    data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/cell_annotation/pancreas/scbert/all.h5ad'
    SEED = 2021
    EPOCHS = 50
    BATCH_SIZE = 3
    GRADIENT_ACCUMULATION = 60
    LEARNING_RATE = 1e-3
    SEQ_LEN = 16907
    VALIDATE_EVERY = 1
    PATIENCE = 10
    UNASSIGN_THRES = 0.0
    CLASS = 7
    POS_EMBED_USING = True
    model_name = 'scbert_pancreas'
    ckpt_dir = './scbert_test'
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ['LOCAL_RANK'])
    print('local rank:', local_rank)
    is_master = local_rank == 0
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    seed_all(SEED + torch.distributed.get_rank())
    train_data, val_data, label_dict = make_dataloader(data_path)
    print(len(train_data.dataset), len(val_data.dataset))
    print(label_dict)
    model = load_model(label_dict)
    train(model, train_data, val_data)
