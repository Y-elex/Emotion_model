# ==============================================================
# dataset.py
# 数据集预处理 & HOG 特征提取
# ==============================================================

import os
import cv2
import numpy as np
from skimage.feature import hog

# ---------------------------
# 参数设置
# ---------------------------
IMAGE_SIZE = (128, 128)   # 统一尺寸
CELL_SIZE = (8, 8)        # HOG cell
BLOCK_SIZE = (2, 2)       # HOG block
N_BINS = 9                # HOG bin 数

# ---------------------------
# Step 1: 图像预处理
# ---------------------------
def preprocess_image(img_path):
    """读取图像 -> 灰度化 -> 直方图均衡化"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE)
    eq = cv2.equalizeHist(gray)
    return eq

# ---------------------------
# Step 2: HOG 特征提取
# ---------------------------
def extract_hog_features(img):
    """计算 HOG 特征向量"""
    features = hog(img,
                   orientations=N_BINS,
                   pixels_per_cell=CELL_SIZE,
                   cells_per_block=BLOCK_SIZE,
                   block_norm="L2-Hys")
    return features

# ---------------------------
# Step 3: 加载数据集
# ---------------------------
def load_split(split_dir):
    """
    从 split 目录加载数据 (train / val / test)
    split_dir: 例如 datasets/AffectNet/train
    """
    X, y = [], []
    class_names = sorted(os.listdir(split_dir))

    for cls_name in class_names:
        cls_path = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        label = int(cls_name)  # 文件夹名即类别编号
        for fname in os.listdir(cls_path):
            img_path = os.path.join(cls_path, fname)
            img = preprocess_image(img_path)
            if img is None:
                continue
            feat = extract_hog_features(img)
            X.append(feat)
            y.append(label)

    return np.array(X), np.array(y)
