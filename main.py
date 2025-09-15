import os
import random
import pandas as pd
from aug import aug_fea_mask, aug_drop_node, aug_fea_drop, aug_fea_dropout
from copy import deepcopy
from logreg import LogReg
def warn(*args, **kwargs):
    pass
from gin import GraphCNN
import promt as Prompt
import warnings
from util import load_data, separate_data
warnings.warn = warn
from arguments import arg_parse
from torch import optim
import label_noise
import numpy as np
import torch
import torch.nn as nn
from inhencedata import hencedata
from topo import TopoGNN
from pseudo_label import generate_pseudo_labels
from torch_geometric.data import Batch
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
from torch_geometric.data import Data
class Trainer:
    def __init__(self, args,device):
        self.args = args
        self.device=device
        self.patience = args.patience
        self.model = Model(args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        sample_input_emb_size = (args.num_gc_layers - 1) * args.hidden_dim
        log_input_dim = sample_input_emb_size+args.out_ph_dim
        self.log = LogReg(log_input_dim, args.num_classes).to(self.device)
        self.prompt = Prompt.SimplePrompt(args.num_node_features).to(device)
        self.xent = nn.CrossEntropyLoss()
        self.topo_gnn = TopoGNN(
            args.hidden_dim,
            num_filtrations=args.num_filtrations,
            filtration_hidden=args.filtration_hidden,
            out_ph_dim=args.out_ph_dim,
            depth=args.depth,
            diagram_type="standard",
            ph_pooling_type="mean",
            dim1=True,
            sig_filtrations=True,
        )
        self.opt = optim.SGD([
            {'params': self.log.parameters(),'weight_decay': args.weight_decay},
            {'params': self.prompt.parameters()},
            {'params': self.topo_gnn.parameters()}
        ], lr=args.lr)

    def custom_to_pyg(self,graph):
        return Data(
            x=graph.node_features,
            edge_index=graph.edge_mat,
            y=graph.label,
            batch=torch.zeros(len(graph.node_features)).long()  # 如果需要
        )
    def train(self, train_graphs, aug1, aug2, epoch_num, patience, dataset_name):
        best = 1e9
        cnt_wait = 0
        graph_copy_2 = deepcopy(train_graphs)
        if aug1 == 'identity':
            graph_aug1 = train_graphs
        elif aug1 == 'node_drop':
            graph_aug1 = aug_drop_node(train_graphs)
        elif aug1 == 'feature_mask':
            graph_aug1 = aug_fea_mask(train_graphs)
        elif aug1 == 'feature_drop':
            graph_aug1 = aug_fea_drop(train_graphs)
        elif aug1 == 'feature_dropout':
            graph_aug1 = aug_fea_dropout(train_graphs)

        if aug2 == 'node_drop':
            graph_aug2 = aug_drop_node(graph_copy_2)
        elif aug2 == 'feature_mask':
            graph_aug2 = aug_fea_mask(graph_copy_2)
        elif aug2 == 'feature_drop':
            graph_aug2 = aug_fea_drop(train_graphs)
        elif aug2 == 'feature_dropout':
            graph_aug2 = aug_fea_dropout(train_graphs)

        print("graph augmentation complete!")

        for i in range(epoch_num):
            self.model.train()
            _,train_embs,_ = self.model(graph_aug1)
            _,train_embs_aug,_ = self.model(graph_aug2)

            loss = self.model.loss_cal(train_embs, train_embs_aug)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if loss == None: continue

            if i % 1== 0:
                print('Epoch {} Loss {:.4f}'.format(i, loss))

                if loss < best:
                    best = loss
                    cnt_wait = 0
                    torch.save(self.model.state_dict(), './savepoint/' + dataset_name + '_model.pkl')
                else:
                    cnt_wait += 1
            if cnt_wait > patience:
                print("Early Stopping!")
                break
    def fine_tune(self, train_graphs, test_graphs, epoch_num, dataset_name, confidence_threshold,
                  batch_size=64):
        self.model.load_state_dict(torch.load('./savepoint/' + self.args.DS + '_model.pkl'))
        print("model load success!")
        best_loss = 1e9
        best_acc=0
        best_t = 0
        cnt_wait = 0
        added_indices = []
        test_accs = []
        test_acc_history = []
        # 应用提示
        for graph in train_graphs:
            graph.node_features = self.prompt.add(graph.node_features)
        for i in range(epoch_num):
            self.model.eval()
            self.log.train()
            self.prompt.train()
            self.topo_gnn.train()
            total_loss = 0
            correct = 0
            total = 0
            j = 0
            while j < len(train_graphs):
                if j + batch_size <= len(train_graphs):
                    batch_graphs = train_graphs[j:j + batch_size]
                    j += batch_size
                else:
                    remaining = len(train_graphs) - j
                    if remaining == 1 and j > 0:
                        batch_graphs = train_graphs[j - batch_size + 1:j + 1]
                        j += 1
                    else:
                        batch_graphs = train_graphs[j:j + remaining]
                        j += remaining
                
                batch_pyg_data = [self.custom_to_pyg(graph) for graph in batch_graphs]
                pooled_h,_, node_embeds = self.model(batch_graphs)
                batch_pyg_data = Batch.from_data_list(batch_pyg_data).to(self.device)
                ph_embedding = self.topo_gnn(batch_pyg_data, node_embeds)
                combined = torch.cat([pooled_h, ph_embedding], dim=1)
                logits = self.log(combined)
                batch_labels = torch.tensor([graph.label for graph in batch_graphs], dtype=torch.long).to(self.device)
                loss = self.xent(logits, batch_labels)
                if best_loss>loss:
                    best_loss=loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item() * len(batch_graphs)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == batch_labels).sum().item()
                total += len(batch_graphs)
            train_acc = correct / total
            print(f'Epoch {i} Train Loss: {total_loss / len(train_graphs):.4f} Train Acc: {train_acc:.4f}')
            test_acc = self.evaluate(test_graphs,batch_size=batch_size)
            test_accs.append(test_acc)

            test_acc_history.append({
                'epoch': i,
                'test_acc': test_acc
            })

            if i % 1== 0 and i>=49:
                pseudo_graphs, added_indices = generate_pseudo_labels(self, test_graphs,
                                                                      confidence_threshold=confidence_threshold,
                                                                      added_indices=added_indices,batch_size=batch_size)
                if pseudo_graphs:
                    train_graphs = train_graphs + pseudo_graphs
                    # 随机打乱训练集顺序
                    random.shuffle(train_graphs)
                    print(f"Added {len(pseudo_graphs)} pseudo-labeled samples to training set.")

            print(f'Epoch {i} Test Acc: {test_acc:.4f}')

            if test_acc >= best_acc:
                best_acc = test_acc
                best_t = i
        if test_accs:
            avg_test_acc = sum(test_accs) / len(test_accs)
            print(f'Final Average Test Acc: {avg_test_acc:.4f}')
        print(f'Best Test Acc: {best_acc:.4f} at Epoch {best_t}')

        print(f'Best Test Acc: {best_acc:.4f} at Epoch {best_t}')

    def evaluate(self, test_graphs, batch_size=128):
        self.model.eval()
        self.log.eval()
        self.prompt.eval()
        self.topo_gnn.eval()
        correct = 0
        total = 0
        temp_test=deepcopy(test_graphs)
        for graph in temp_test:
            graph.node_features = self.prompt.add(graph.node_features)
        with torch.no_grad():
            i = 0
            while i < len(temp_test):
                if i + batch_size <= len(temp_test):
                    batch_graphs = temp_test[i:i + batch_size]
                    i += batch_size
                else:
                    remaining = len(temp_test) - i
                    if remaining == 1 and i > 0:
                        batch_graphs = temp_test[i - batch_size + 1:i + 1]
                        i += 1
                    else:
                        batch_graphs = temp_test[i:i + remaining]
                        i += remaining
                if len(batch_graphs) <= 1:
                    continue
            
                batch_pyg_data = [self.custom_to_pyg(graph) for graph in batch_graphs]

                # 获取 GNN 嵌入
                pooled_h,_, node_embeds = self.model(batch_graphs)

                # 合并批量的 PyG 数据
                from torch_geometric.data import Batch
                batch_pyg_data = Batch.from_data_list(batch_pyg_data).to(self.device)

                # 获取拓扑特征
                ph_embedding = self.topo_gnn(batch_pyg_data, node_embeds)


                # 合并特征

                combined = torch.cat([pooled_h, ph_embedding], dim=1)
                # combined = pooled_h
                # 使用 LogReg 进行分类
                logits = self.log(combined)
                batch_labels = torch.tensor([graph.label for graph in batch_graphs], dtype=torch.long).to(self.device)
                _, predicted = torch.max(logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        return correct / total

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.T = args.t
        self.hid = args.hidden_dim
        self.device = device
        self.gin = GraphCNN(input_dim=args.num_node_features, use_select_sim=args.use_select_sim, num_layers=args.num_gc_layers,
                            hidden_dim=args.hidden_dim).to(self.device)  # .cuda()
        self.sample_input_emb_size =(args.num_gc_layers - 1) * args.hidden_dim
        self.proj_head = nn.Sequential(nn.Linear(self.sample_input_emb_size, self.hid), nn.ReLU(inplace=True),
                                       nn.Linear(self.hid, self.hid))  # whether to use the batchnorm1d?


    def forward(self, batch_graph):
        output_embeds, node_embeds, Adj_block_idx, _, _,x = self.gin(batch_graph)
        pooled_h = torch.cat(output_embeds[1:], -1)
        pooled_h_p = self.proj_head(pooled_h)
        return pooled_h,pooled_h_p,x

    def loss_cal(self, x, x_aug):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / self.T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
if __name__ == '__main__':
    for i in range(0,10):
        args = arg_parse()
        args.fold_idx = i
        setup_seed(args.seed)
        epoch=1000
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        graphs, num_classes,dataset_num_features = load_data(args.DS,args.degree_as_tag)
        args.num_node_features = dataset_num_features
        args.num_classes=num_classes

        train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
        ptb = args.ptb
        noise_type = args.noise
        train_labels = [graph.label for graph in train_graphs]
        train_labels = np.array(train_labels)

        noise_y, P = label_noise.noisify_p(train_labels, num_classes, ptb, noise_type, args.seed)
        for i in range(len(train_graphs)):
            train_graphs[i].label = noise_y[i]
        print("*"*60)
        print(len(train_graphs))
        graph_train = deepcopy(train_graphs)
        train_graphs=hencedata(train_graphs,200)
        print(len(train_graphs))
        model=Trainer(args,device)
        model.train(train_graphs=train_graphs,aug1=args.aug1,aug2=args.aug2,epoch_num=epoch,patience=args.patience,dataset_name=args.DS)
        # 训练结束

        model.fine_tune(graph_train, test_graphs, epoch_num=350,dataset_name=args.DS,
                           confidence_threshold=0.8)



