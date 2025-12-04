# model.py
import torch
import torch.nn as nn

class FuseMLP(nn.Module):
    def __init__(self, bow_dim, cnn_dim=2048, num_classes=8, hidden=[1024,256], dropout=0.5):
        super(FuseMLP, self).__init__()
        in_dim = bow_dim + cnn_dim
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, bow, cnn):
        x = torch.cat([bow, cnn], dim=1)
        return self.net(x)
