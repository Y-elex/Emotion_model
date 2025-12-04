# ==============================================================
# model.py
# HOG 特征分类器：全连接层 + Softmax
# ==============================================================

import torch
import torch.nn as nn

class ExpressionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ExpressionClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # CrossEntropyLoss 内部自带 softmax
