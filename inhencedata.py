import json
import math
import random
import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from numpy.random import RandomState
from collections import defaultdict
from torch_geometric.utils import to_dense_adj, dense_to_sparse
class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def hencedata(train_graphs,gen_train_num):
    generate_train_graphs = []
    fir, sec = np.random.randint(low=0, high=len(train_graphs), size=(2,gen_train_num))
    for i in range(gen_train_num):
        if i % 64 == 0:
            lam = np.random.beta(0.5, 0.5)
        lam = max(lam, 1 - lam)
        # emb1 = self.train_graphs[fir[i]].node_features / self.train_graphs[fir[i]].norm(dim=1)
        match = train_graphs[fir[i]].node_features @ train_graphs[sec[i]].node_features.T
        normalized_match = F.softmax(match, dim=0)

        mixed_adj = lam * to_dense_adj(train_graphs[fir[i]].edge_mat)[0].double() + (
                    1 - lam) * normalized_match.double() @ to_dense_adj(train_graphs[sec[i]].edge_mat)[
                        0].double() @ normalized_match.double().T
        mixed_adj[mixed_adj < 0.1] = 0
        mixed_x = lam * train_graphs[fir[i]].node_features + (1 - lam) * normalized_match.float() @ \
                  train_graphs[sec[i]].node_features

        edge_index, _ = dense_to_sparse(mixed_adj)
        edges = [(x, y) for x, y in zip(edge_index[0].tolist(), edge_index[1].tolist())]
        g = nx.Graph()
        g.add_edges_from(edges)
        g.add_nodes_from(list(range(edge_index.max() + 1)))
        G = S2VGraph(g, -1)
        G.edge_mat = edge_index
        G.node_features = mixed_x
        generate_train_graphs.append(G)
    print("generate yes")
    print("generate len is ", len(generate_train_graphs))
    print("before the number of train graphs is ", len(train_graphs))
    train_graphs.extend(generate_train_graphs)
    print("after the number of train graphs is ", len(train_graphs))
    return train_graphs


def generate_test_mixup_interpolate_labels(test_graphs, gen_per_class=20):
    from collections import defaultdict
    import numpy as np
    import torch.nn.functional as F
    from torch_geometric.utils import to_dense_adj, dense_to_sparse
    import networkx as nx

    # 按类别分组
    class2graphs = defaultdict(list)
    for g in test_graphs:
        class2graphs[g.label].append(g)

    generated_graphs = []

    for label, graphs in class2graphs.items():
        n = len(graphs)
        if n < 2:
            continue  # 少于2个图无法mixup
        fir, sec = np.random.randint(0, n, size=(2, gen_per_class))
        for i in range(gen_per_class):
            lam = np.random.beta(0.5, 0.5)
            lam = max(lam, 1 - lam)
            g1, g2 = graphs[fir[i]], graphs[sec[i]]
            match = g1.node_features @ g2.node_features.T
            norm_match = F.softmax(match, dim=0)
            adj1 = to_dense_adj(g1.edge_mat)[0].double()
            adj2 = to_dense_adj(g2.edge_mat)[0].double()
            mixed_adj = lam * adj1 + (1 - lam) * norm_match.double() @ adj2 @ norm_match.double().T
            mixed_adj[mixed_adj < 0.1] = 0
            mixed_x = lam * g1.node_features + (1 - lam) * norm_match.float() @ g2.node_features
            interpolated_label = lam * g1.label + (1 - lam) * g2.label
            edge_index, _ = dense_to_sparse(mixed_adj)
            edges = [(x, y) for x, y in zip(edge_index[0].tolist(), edge_index[1].tolist())]
            g = nx.Graph()
            g.add_edges_from(edges)
            g.add_nodes_from(range(edge_index.max().item() + 1))

            G = S2VGraph(g, label=interpolated_label)
            G.edge_mat = edge_index
            G.node_features = mixed_x
            generated_graphs.append(G)
    test_graphs.extend(generated_graphs)
    return test_graphs