import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes, hidden_dim=64):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(ft_in, hidden_dim)
        self.bn = nn.BatchNorm1d(ft_in)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim, nb_classes)

    def forward(self, seq):
        x = self.bn(seq)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        ret = self.fc2(x)
        return ret