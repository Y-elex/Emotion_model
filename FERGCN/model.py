import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: batch x N x F, adj: batch x N x N
        h = torch.bmm(adj, x)  # 邻接矩阵乘特征
        h = self.linear(h)
        h = F.relu(h)
        return h

class FERGCN(nn.Module):
    def __init__(self, in_features=2, hidden=64, num_classes=8):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        # 全局平均池化
        h = h.mean(dim=1)
        out = self.fc(h)
        return out
