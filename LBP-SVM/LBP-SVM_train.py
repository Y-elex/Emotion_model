import os
import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from cuml.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# ========================
# 参数设置
# ========================
data = 'AffectNet'  # 数据集名称，可选 'CK+', 'JAFFE', 'RAF-DB'
data_dir = f"../datasets/{data}"
radius = 1
n_points = 8 * radius
METHOD = 'uniform'
orb_features_dim = 32   # ORB 描述子平均池化后长度固定 32
n_jobs = -1  # CPU 并行线程数

# ========================
# 特征提取函数
# ========================
def extract_lbp(image_gray):
    lbp = local_binary_pattern(image_gray, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def extract_orb(image_gray, n_features=200):
    # 每次调用创建 ORB 对象，避免 joblib pickle 错误
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    if descriptors is None:
        return np.zeros(orb_features_dim)
    return descriptors.mean(axis=0)

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (96, 96))
    lbp_feat = extract_lbp(img)
    orb_feat = extract_orb(img)
    feat = np.hstack([lbp_feat, orb_feat])
    return feat

# ========================
# 数据加载函数（并行化）
# ========================
def load_data(split="train"):
    X, y = [], []
    split_dir = os.path.join(data_dir, split)
    classes = sorted(os.listdir(split_dir))
    for label in classes:
        class_dir = os.path.join(split_dir, label)
        img_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
        feats = Parallel(n_jobs=n_jobs)(
            delayed(extract_features)(p) for p in tqdm(img_files, desc=f"{split}-{label}")
        )
        feats = [f for f in feats if f is not None]
        X.extend(feats)
        y.extend([int(label)] * len(feats))
    return np.array(X), np.array(y)

# ========================
# 训练 & 验证
# ========================
def main():
    print("加载训练数据...")
    X_train, y_train = load_data("train")
    print("加载验证数据...")
    X_val, y_val = load_data("val")
    print(f"训练样本: {X_train.shape}, 验证样本: {X_val.shape}")

    # 使用 GPU 加速训练
    clf = LinearSVC(C=1.0, max_iter=5000)
    clf.fit(X_train, y_train)

    # 训练集预测
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"训练集准确率: {train_acc:.4f}")

    # 验证
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"验证集准确率: {acc:.4f}")

    # 保存模型
    joblib.dump(clf, f"lbp_orb_linearSVC_{data}_model.pkl")
    print("模型已保存: lbp_orb_linearSVC_model.pkl")

    # 保存日志
    df_log = pd.DataFrame({"Accuracy": [acc]})
    df_log.to_excel(f"training_log_{data}.xlsx", index=False)
    print("日志已保存: training_log.xlsx")

    # 画图
    plt.figure()
    plt.bar(["Validation"], [acc])
    plt.ylabel("Accuracy")
    plt.title("LBP+ORB Facial Expression Recognition")
    plt.savefig(f"accuracy_{data}_plot.png")
    print("准确率图已保存: accuracy_plot.png")

if __name__ == "__main__":
    main()
