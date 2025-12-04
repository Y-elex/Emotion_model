import os
import torch
import torch.utils.data as Data
import numpy as np

class LandmarkGraphDataset(Data.Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []

        for label in sorted(os.listdir(root_dir)):
            folder = os.path.join(root_dir, label)
            if not os.path.isdir(folder):
                continue
            for npy_name in os.listdir(folder):
                self.samples.append(os.path.join(folder, npy_name))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kp_path = self.samples[idx]   # ⭐ 记录路径
        keypoints = np.load(kp_path)
        label = self.labels[idx]

        # 构建简单 KNN 邻接矩阵
        N = keypoints.shape[0]
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            dists = np.linalg.norm(keypoints - keypoints[i], axis=1)
            knn_idx = np.argsort(dists)[1:9]  # 最近 8 个
            adj[i, knn_idx] = 1

        node_features = torch.tensor(keypoints, dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.float)

        # ⭐ 多返回一个路径
        return node_features, adj, label, kp_path
