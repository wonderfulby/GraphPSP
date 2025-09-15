import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge

from layers.rephine_layer import RephineLayer



class TopoGNN(nn.ModuleList):
    def __init__(
        self,
        hidden_dim,
        num_filtrations,
        filtration_hidden,
        out_ph_dim,
        depth,
        diagram_type="standard",
        ph_pooling_type="mean",
        dim1=True,
        sig_filtrations=True,
    ):
        super().__init__()
        self.depth = depth
        topo_layers = []
        self.ph_pooling_type = ph_pooling_type
        for i in range(depth):
            topo = RephineLayer(
            n_features=hidden_dim,  # 输入特征维度
            n_filtrations=num_filtrations,  # 过滤函数数量
            filtration_hidden=filtration_hidden,
            out_dim=out_ph_dim,  # 拓扑特征输出维度
            diagram_type=diagram_type,  # "rephine"或"standard"
            dim1=dim1,  # 是否处理1维拓扑特征
            sig_filtrations=sig_filtrations  # 是否使用Sigmoid过滤函数
            )
            topo_layers.append(topo)

        self.ph_layers = nn.ModuleList(topo_layers)# 注册拓扑层

        if self.ph_pooling_type != "mean":
            self.jump = JumpingKnowledge(mode=self.ph_pooling_type)

    def forward(self, data,x_ph):
        ph_vectors = []
        for i in range(0,self.depth):
            x = x_ph[i]
            ph_vectors += [self.ph_layers[i](x, data)]
        if self.ph_pooling_type == "mean":
            ph_embedding = torch.stack(ph_vectors).mean(dim=0)
        else:
            ph_embedding = self.jump(ph_vectors)

        return  ph_embedding

