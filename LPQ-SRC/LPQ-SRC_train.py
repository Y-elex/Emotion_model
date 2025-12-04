import os
import cv2
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# -----------------------------
# LPQ 特征提取函数
# -----------------------------
def lpq(img, win_size=3, freq_est=1.0):
    rho = 0.90
    STFT_alpha = freq_est
    img = np.float32(img)

    x = np.arange(-win_size//2 + 1, win_size//2 + 1)
    w0 = (1.0 * x / win_size) * STFT_alpha
    w1 = np.exp(-2j * np.pi * w0)

    filter1 = np.real(np.outer(w1, np.ones_like(w1)))
    filter2 = np.imag(np.outer(w1, np.ones_like(w1)))
    filter3 = np.real(np.outer(np.ones_like(w1), w1))
    filter4 = np.imag(np.outer(np.ones_like(w1), w1))

    f1 = cv2.filter2D(img, -1, filter1, borderType=cv2.BORDER_REFLECT)
    f2 = cv2.filter2D(img, -1, filter2, borderType=cv2.BORDER_REFLECT)
    f3 = cv2.filter2D(img, -1, filter3, borderType=cv2.BORDER_REFLECT)
    f4 = cv2.filter2D(img, -1, filter4, borderType=cv2.BORDER_REFLECT)

    LPQdesc = (f1 > 0).astype(np.uint8) + \
              ((f2 > 0).astype(np.uint8) << 1) + \
              ((f3 > 0).astype(np.uint8) << 2) + \
              ((f4 > 0).astype(np.uint8) << 3)

    hist, _ = np.histogram(LPQdesc, bins=256, range=(0, 256), density=True)
    return hist

# -----------------------------
# 数据加载
# -----------------------------
def load_dataset(root_dir, img_size=120):
    X, y = [], []
    for label in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for img_file in tqdm(os.listdir(class_dir), desc=f"Loading {label}"):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            feat = lpq(img)
            X.append(feat)
            y.append(int(label))
    return np.array(X), np.array(y)

# -----------------------------
# SRC 分类函数
# -----------------------------
def src_classify(trainX, trainY, testX):
    preds = []
    for x in tqdm(testX, desc="SRC Testing"):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=30)
        omp.fit(trainX.T, x)
        coef = omp.coef_

        residuals = []
        for c in np.unique(trainY):
            idx = np.where(trainY == c)[0]
            coef_c = np.zeros_like(coef)
            coef_c[idx] = coef[idx]
            x_c = trainX.T @ coef_c
            residuals.append(np.linalg.norm(x - x_c))
        preds.append(np.argmin(residuals))
    return np.array(preds)

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    trainX, trainY = load_dataset("datasets/AffectNet/train")
    valX, valY = load_dataset("datasets/AffectNet/val")

    # 标准化
    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    valX = scaler.transform(valX)

    # ✅ 训练集准确率
    train_preds = src_classify(trainX, trainY, trainX)
    train_acc = accuracy_score(trainY, train_preds)
    print("SRC Training Accuracy: {:.2f}%".format(train_acc * 100))

    # ✅ 验证集准确率
    val_preds = src_classify(trainX, trainY, valX)
    val_acc = accuracy_score(valY, val_preds)
    print("SRC Validation Accuracy: {:.2f}%".format(val_acc * 100))

    # 保存字典
    np.savez("src_dictionary.npz", trainX=trainX, trainY=trainY)
    print("✅ 已保存 SRC 字典: src_dictionary.npz")
