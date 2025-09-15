import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    def __init__(
        self, in_features, out_features, activation, batch_norm, residual=True
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()

        self.residual = residual
        self.conv = GCNConv(in_features, out_features, add_self_loops=False)
#gcn内部操作
    def forward(self, x, edge_index):
        # [A] 消息传递与聚合
        h = self.conv(x, edge_index)
        # [B] 批归一化
        h = self.batchnorm(h)
        # [C] 激活函数
        h = self.activation(h)
        # [D] 残差连接
        if self.residual:
            h = h + x
        return h
