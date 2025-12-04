import cv2
import numpy as np
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit
import joblib

# 加载训练好的模型字典
trainX = joblib.load("trainX.pkl")  # 训练集特征矩阵
trainY = joblib.load("trainY.pkl")  # 训练集标签

# LPQ 参数（需和训练时一致）
win_size = 11
rho = 0.90
freq_estim = 1

def lpq(img, win_size=win_size, rho=rho, freq_estim=freq_estim):
    """
    对单张图像提取旋转不变 LPQ 特征
    img: 灰度图
    返回: LPQ 直方图特征向量
    """
    # 确保图像为 float32
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    
    # 这里使用你已有的 lpq() 实现或改写，返回特征向量
    # 例如 lpq_vector = your_lpq_function(img)
    # 这里简化用假设函数
    lpq_vector = lpq_vector_from_image(img, win_size, rho, freq_estim)  # 你需要定义
    return lpq_vector

def predict_expression(img_path):
    """
    输入单张图片路径，输出预测表情标签
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    feat = lpq(img)
    
    # 标准化（训练时用的同一方式）
    feat = feat.astype(np.float32)
    feat = (feat - feat.mean()) / (feat.std() + 1e-8)
    
    # SRC 分类
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=30)
    omp.fit(trainX.T, feat)
    coef = omp.coef_
    
    # 计算各类别残差
    residuals = []
    classes = np.unique(trainY)
    for c in classes:
        idx = np.where(trainY == c)[0]
        x_c = trainX[:, idx]
        coef_c = coef[idx]
        recon = x_c @ coef_c
        res = np.linalg.norm(feat - recon)
        residuals.append(res)
    
    pred_class = classes[np.argmin(residuals)]
    return int(pred_class)

# 示例
img_path = "datasets/AffectNet/test/0/img001.jpg"
pred_label = predict_expression(img_path)
print(f"Predicted label: {pred_label}")
