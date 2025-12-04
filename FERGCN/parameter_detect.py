import torch
import torch.nn as nn
from model import FERGCN  # 导入你的 GCN 模型

def count_flops_fergcn(model, batch_size=1, N=68, F_in=2):
    """
    估算 FERGCN 模型 FLOPs，并打印每层占比
    batch_size: 批量大小
    N: 节点数（关键点数）
    F_in: 输入特征维度
    """
    F_hidden = model.gcn1.linear.out_features
    F_out = model.fc.out_features

    # GCN Layer 1: bmm(adj, x) + Linear
    gcn1_bmm = batch_size * N * N * F_in * 2
    gcn1_linear = batch_size * N * F_in * F_hidden * 2
    gcn1_total = gcn1_bmm + gcn1_linear

    # GCN Layer 2: bmm(adj, x) + Linear
    gcn2_bmm = batch_size * N * N * F_hidden * 2
    gcn2_linear = batch_size * N * F_hidden * F_hidden * 2
    gcn2_total = gcn2_bmm + gcn2_linear

    # 全局平均池化: 忽略
    pool_flops = 0

    # 最后一层全连接
    fc_flops = batch_size * F_hidden * F_out * 2

    total_flops = gcn1_total + gcn2_total + pool_flops + fc_flops

    # 打印每层 FLOPs 和占比
    print("FLOPs 估算（乘加次数）:")
    print(f"  GCN Layer 1: {gcn1_total:,} ({gcn1_total/total_flops*100:.2f}%)")
    print(f"  GCN Layer 2: {gcn2_total:,} ({gcn2_total/total_flops*100:.2f}%)")
    print(f"  FC Layer:     {fc_flops:,} ({fc_flops/total_flops*100:.2f}%)")
    print(f"  Total FLOPs:  {total_flops:,}")
    return total_flops

if __name__ == "__main__":
    # 初始化模型
    model = FERGCN(in_features=2, hidden=64, num_classes=8)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

    # 估算 FLOPs
    batch_size = 1
    N = 68
    F_in = 2
    total_flops = count_flops_fergcn(model, batch_size=batch_size, N=N, F_in=F_in)

