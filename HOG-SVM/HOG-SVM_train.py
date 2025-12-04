import os
import cv2
import cupy as cp
import numpy as np
from cuml.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============ 参数设置 ============
data_dir = "/root/autodl-tmp/datasets/AffectNet/train"
image_size = (128, 128)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# ============ HOG 初始化 ============
hog = cv2.HOGDescriptor(
    _winSize=(image_size[1] // pixels_per_cell[1] * pixels_per_cell[1],
              image_size[0] // pixels_per_cell[0] * pixels_per_cell[0]),
    _blockSize=(cells_per_block[1] * pixels_per_cell[1],
                cells_per_block[0] * pixels_per_cell[0]),
    _blockStride=(pixels_per_cell[1], pixels_per_cell[0]),
    _cellSize=(pixels_per_cell[1], pixels_per_cell[0]),
    _nbins=orientations
)

# ============ HOG 特征提取 ============
def extract_hog(image_gray):
    img = cv2.resize(image_gray, image_size)
    return hog.compute(img).flatten()

# ============ 数据加载 ============
def load_data(data_path):
    X, y = [], []
    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    for label_idx, label in enumerate(classes):
        class_dir = os.path.join(data_path, label)
        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = extract_hog(img)
            X.append(feat)
            y.append(label_idx)
    return cp.asarray(X, dtype=cp.float32), cp.asarray(y, dtype=cp.int32), classes

# ============ 主函数 ============
def main():
    print("加载数据...")
    X, y, classes = load_data(data_dir)
    print(f"样本数量: {X.shape[0]}, 特征维度: {X.shape[1]}, 类别: {classes}")

    # 打乱数据
    idx = cp.random.permutation(X.shape[0])
    X, y = X[idx], y[idx]

    # 划分训练/验证
    split = int(0.8 * X.shape[0])
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("训练 cuML SVM...")
    clf = SVC(kernel='rbf', C=10, gamma='scale')
    clf.fit(X_train, y_train)

    print("预测验证集...")
    y_pred = clf.predict(X_val)

    # 计算准确率
    acc = cp.mean(y_pred == y_val)
    print(f"验证集准确率: {acc:.4f}")

    # 混淆矩阵（用 numpy 绘图）
    y_pred_np = cp.asnumpy(y_pred)
    y_val_np = cp.asnumpy(y_val)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_val_np, y_pred_np):
        cm[t, p] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix_gpu.png")
    print("混淆矩阵已保存: confusion_matrix_gpu.png")

    # ============ 保存模型 ============
    model_path = "hog_svm_gpu_model.pkl"
    joblib.dump(clf, model_path)
    print(f"模型已保存: {model_path}")

if __name__ == "__main__":
    main()
