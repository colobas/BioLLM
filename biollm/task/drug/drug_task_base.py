import argparse
import random, os, sys
import numpy as np
import csv
import time
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
sys.path.append('/home/share/huadjyin/home/s_qiuping1/hanyuxuan')   ####################
from biollm.base.bio_task import BioTask
from biollm.model.drug import PyTorchMultiSourceGCNModel


class DrugTask(BioTask):
    def __init__(self, config_file):
        super(DrugTask, self).__init__(config_file)
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_id
        self.device = self.args.device
        self.TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                          "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                          "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                          "STAD", "THCA", 'COAD/READ']

    def MetadataGenerate(self, Drug_info_file, Cell_line_info_file, Genomic_mutation_file, Drug_feature_file,
                         Gene_expression_file, Methylation_file, filtered):
        # drug_id --> pubchem_id
        with open(Drug_info_file, 'r') as f:
            reader = csv.reader(f)
            rows = [item for item in reader]
            drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

        # map cellline --> cancer type
        cellline2cancertype = {}
        with open(Cell_line_info_file) as f:
            for line in f.readlines()[1:]:
                cellline_id = line.split('\t')[1]
                TCGA_label = line.strip().split('\t')[-1]
                # if TCGA_label in TCGA_label_set:
                cellline2cancertype[cellline_id] = TCGA_label

        # load demap cell lines genomic mutation features
        mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
        cell_line_id_set = list(mutation_feature.index)

        # load drug features
        drug_pubchem_id_set = []
        drug_feature = {}
        for each in os.listdir(Drug_feature_file):
            drug_pubchem_id_set.append(each.split('.')[0])
            feat_mat, adj_list, degree_list = torch.load('%s/%s' % (Drug_feature_file, each))
            drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
        assert len(drug_pubchem_id_set) == len(drug_feature.values())

        # load gene expression features
        gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])

        # only keep overlapped cell lines
        mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]

        # load methylation
        methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])
        assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
        experiment_data = pd.read_csv(self.args.cancer_response_exp_file, sep=',', header=0, index_col=[0])
        # filter experiment data
        drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
        experiment_data_filtered = experiment_data.loc[drug_match_list]

        data_idx = []
        for each_drug in experiment_data_filtered.index:
            for each_cellline in experiment_data_filtered.columns:
                pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
                if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                    if not np.isnan(experiment_data_filtered.loc[
                                        each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                        ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                        data_idx.append((each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))
        nb_celllines = len(set([item[0] for item in data_idx]))
        nb_drugs = len(set([item[1] for item in data_idx]))
        print(
            '%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), nb_celllines, nb_drugs))
        return mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx

    def DataSplit(self, data_idx, ratio=0.95):
        data_train_idx, data_test_idx = [], []
        for each_type in self.TCGA_label_set:
            data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
            train_list = random.sample(data_subtype_idx, int(ratio * len(data_subtype_idx)))
            test_list = [item for item in data_subtype_idx if item not in train_list]
            data_train_idx += train_list
            data_test_idx += test_list
        print('Data split.')
        return data_train_idx, data_test_idx

    def NormalizeAdj(self, adj):
        adj = adj + torch.eye(adj.shape[0], device=self.device)
        d = torch.diag(torch.pow(adj.sum(1), -0.5))
        a_norm = adj.mm(d).t().mm(d)
        return a_norm

    def random_adjacency_matrix(self, n):
        matrix = torch.randint(0, 2, (n, n), device=self.device)
        # No vertex connects to itself
        matrix.fill_diagonal_(0)
        # If i is connected to j, j is connected to i
        matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
        return matrix

    # calculate feature matrix and adjacency matrix
    def CalculateGraphFeat(self, feat_mat, adj_list):
        assert feat_mat.shape[0] == len(adj_list)
        feat = torch.zeros((self.args.max_atoms, feat_mat.shape[-1]), dtype=torch.float32, device=self.device)
        adj_mat = torch.zeros((self.args.max_atoms, self.args.max_atoms), dtype=torch.float32, device=self.device)
        if self.args.israndom:
            feat = torch.rand(self.args.max_atoms, feat_mat.shape[-1], device=self.device)
            adj_mat[feat_mat.shape[0]:, feat_mat.shape[0]:] = self.random_adjacency_matrix(self.args.max_atoms - feat_mat.shape[0])
        feat[:feat_mat.shape[0], :] = torch.from_numpy(feat_mat).to(self.device)
        for i in range(len(adj_list)):
            nodes = adj_list[i]
            for each in nodes:
                adj_mat[i, int(each)] = 1
        assert torch.allclose(adj_mat, adj_mat.T)
        adj_ = adj_mat[:len(adj_list), :len(adj_list)]
        adj_2 = adj_mat[len(adj_list):, len(adj_list):]
        norm_adj_ = self.NormalizeAdj(adj_)
        norm_adj_2 = self.NormalizeAdj(adj_2)
        adj_mat[:len(adj_list), :len(adj_list)] = norm_adj_
        adj_mat[len(adj_list):, len(adj_list):] = norm_adj_2
        return [feat, adj_mat]

    def FeatureExtract(self, data_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature):
        cancer_type_list = []
        nb_instance = len(data_idx)
        nb_mutation_feature = mutation_feature.shape[1]
        nb_gexpr_features = gexpr_feature.shape[1]
        nb_methylation_features = methylation_feature.shape[1]

        # data initialization
        drug_data = [[] for item in range(nb_instance)]
        mutation_data = torch.zeros((nb_instance, 1, nb_mutation_feature, 1), dtype=torch.float32, device=self.device)
        gexpr_data = torch.zeros((nb_instance, nb_gexpr_features), dtype=torch.float32, device=self.device)
        methylation_data = torch.zeros((nb_instance, nb_methylation_features), dtype=torch.float32, device=self.device)
        target = torch.zeros(nb_instance, dtype=torch.float32, device=self.device)

        print('Feature Extracting...')
        for idx in tqdm(range(nb_instance)):
            cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]
            # modify
            feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]
            # fill drug data,padding to the same size with zeros
            drug_data[idx] = self.CalculateGraphFeat(feat_mat, adj_list)
            # randomlize X A
            mutation_data[idx, 0, :, 0] = torch.from_numpy(mutation_feature.loc[cell_line_id].values).float().to(self.device)
            gexpr_data[idx, :] = torch.from_numpy(gexpr_feature.loc[cell_line_id].values).float().to(self.device)
            methylation_data[idx, :] = torch.from_numpy(methylation_feature.loc[cell_line_id].values).float().to(self.device)
            target[idx] = ln_IC50
            cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
        return drug_data, mutation_data, gexpr_data, methylation_data, target, cancer_type_list

    # 训练
    def train(self, model, dataloader, validation_data, optimizer, nb_epoch):
        patience = 10
        best_pcc = -np.Inf  # record best pcc
        best_epoch = 0  # record epoch of best pcc
        counter = 0
        # 对每一轮训练
        for epoch in range(0, nb_epoch):
            loss_list = []
            t = time.time()
            for ii, data_ in enumerate(dataloader):
                model.train()  # switch on batch normalization and dropout
                data_ = [dat.to(self.device) for dat in data_]
                X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data, X_methylation_data, Y_train = data_
                output = model(X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data,
                               X_methylation_data)  # calculate ouput
                output = output.squeeze(-1)
                loss = F.mse_loss(output, Y_train)
                pcc = torch.corrcoef(torch.stack((output, Y_train)))[0, 1]  # pcc
                # spear, _ = spearmanr(output.detach().cpu().numpy(), Y_train.detach().cpu().numpy())  # spearman
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ii % 500 == 0:
                    print(f"[INFO] epoch {epoch} batch {ii}:  loss {loss.item():.4f}  pcc {pcc:.4f}")

            loss_test, pcc_test, spearman_test = self.test(model, validation_data)  # loss pcc scc
            epoch_loss = sum(loss_list) / len(loss_list)
            torch.save(model.state_dict(), f'{self.args.save_path}/deepcdr/{epoch}.pth')
            print(f'[INFO] epoch {epoch}: epoch average loss {epoch_loss:.4f}')
            print(f'[INFO] validation data: loss {loss_test:.4f}    pcc {pcc_test:.4f} spearman {spearman_test:.4f}')
            print('=========================================================')
            # early stop
            if pcc_test > best_pcc:
                best_pcc = pcc_test
                best_epoch = epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:  # early stop
                    print(f'[Early stop] best epoch {best_epoch}   best pcc {best_pcc: .4f}')
                    model.load_state_dict(
                        torch.load(f'{self.args.save_path}/deepcdr/{best_epoch}.pth', map_location=self.device))
                    print(f'Load best model from {self.args.save_path}/deepcdr/{best_epoch}.pth')
                    loss, pcc, scc = self.test(model, validation_data)
                    print(f'loss {loss:.4f}   pcc {pcc:.4f}   scc {scc:.4f}')
                    break

    # 测试
    def test(self, model, validation_data):
        print('Testing...')
        model.eval()  # switch off batch normalization and dropout
        with torch.no_grad():
            validation_data[0] = [dat.to(self.device) for dat in validation_data[0]]
            validation_data[1] = validation_data[1].to(self.device)
            X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test = validation_data[0]
            Y_test = validation_data[1]  # 验证集的标签
            test_data = TensorDataset(X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test,
                                      X_gexpr_data_test, X_methylation_data_test)
            test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
            # predict output with batch size 64
            output_test = torch.empty((0,), device=self.device)  # create an empty tensor
            for ii, data_ in enumerate(test_dataloader):
                X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data, X_methylation_data = data_
                output = model(X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data, X_methylation_data)
                output = output.squeeze(-1)
                output_test = torch.cat((output_test, output), dim=0)
            loss_test = F.mse_loss(output_test, Y_test)
            pcc = torch.corrcoef(torch.stack((output_test, Y_test)))[0, 1]  # pcc
            spearman_test, _ = spearmanr(output_test.detach().cpu().numpy(), Y_test.detach().cpu().numpy())  # spearman
        return loss_test.item(), pcc.item(), spearman_test

    def run(self):
        random.seed(self.args.seed)
        # data_idx: (cell_line, drug, ln_IC50, cell_type)  example: ('ACH-000534', '9907093', 4.358842, 'DLBC')
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = self.MetadataGenerate(self.args.drug_info_file,self.args.cell_line_info_file,self.args.genomic_mutation_file,
                                                                                            self.args.drug_feature_file,self.args.gene_expression_file,self.args.methylation_file,False)

        # train / test
        data_train_idx, data_test_idx = self.DataSplit(data_idx)
        # extract feature
        X_drug_data_train, X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train, Y_train, cancer_type_train_list = self.FeatureExtract(
            data_train_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
        X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list = self.FeatureExtract(
            data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
        # extract feature matrix and adjacency matrix
        X_drug_feat_data_train = torch.stack([item[0] for item in X_drug_data_train])  # # nb_instance * Max_stom * feat_dim
        X_drug_adj_data_train = torch.stack([item[1] for item in X_drug_data_train])  # nb_instance * Max_stom * Max_stom
        # extract feature matrix and adjacency matrix
        X_drug_feat_data_test = torch.stack([item[0] for item in X_drug_data_test])  # 维度：nb_instance * Max_stom * feat_dim
        X_drug_adj_data_test = torch.stack([item[1] for item in X_drug_data_test])  # 维度：nb_instance * Max_stom * Max_stom

        # validation data
        validation_data = [[X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test],Y_test]

        # dataLoader
        train_data = TensorDataset(X_drug_feat_data_train, X_drug_adj_data_train, X_mutation_data_train, X_gexpr_data_train,
                                   X_methylation_data_train, Y_train)
        dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

        # model initialization
        model = PyTorchMultiSourceGCNModel(drug_input_dim=X_drug_data_train[0][0].shape[-1], drug_hidden_dim=256, drug_concate_before_dim=100,
                                           mutation_input_dim=X_mutation_data_train.shape[-2], mutation_hidden_dim=256, mutation_concate_before_dim=100,
                                           gexpr_input_dim=X_gexpr_data_train.shape[-1], gexpr_hidden_dim=256, gexpr_concate_before_dim=100,
                                           methy_input_dim=X_methylation_data_train.shape[-1], methy_hidden_dim=256, methy_concate_before_dim=100,
                                           output_dim=300, units_list=self.args.unit_list, use_mut=self.args.use_mut,
                                           use_gexp=self.args.use_gexp, use_methy=self.args.use_methy,
                                           regr=True, use_relu=self.args.use_relu,
                                           use_bn=self.args.use_bn, use_GMP=self.args.use_GMP
                                           ).to(self.device)

        # GPU or CPU
        print('Device Is %s' % self.device)

        # model train / test
        optimizer = Adam(model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0, amsgrad=False)
        if self.args.mode == 'train':
            print('Begin training...')
            self.train(model, dataloader, validation_data, optimizer, nb_epoch=100)
        if self.args.mode == 'test':
            model = torch.load('%s/deepcdr_model.pt' % self.args.save_path)
            loss_test, pcc_test, spearman_test = self.test(model, validation_data)
            print(f'loss {loss_test: .4f}    pcc {pcc_test: .4f}    spearman {spearman_test: .4f}')



if __name__ == "__main__":
    config_file = '../../config/drug/drug.toml'
    obj = DrugTask(config_file)
    obj.run()

