from sklearn.base import accuracy_score
from FER_HOG.train import X_train
import os, cv2, joblib, numpy as np, cupy as cp
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from cuml.svm import SVC

# --------------------
# 参数
# --------------------
train_dir = "/root/autodl-tmp/datasets/AffectNet/train"
val_dir   = "/root/autodl-tmp/datasets/AffectNet/val"   # 若不存在则从 train 切分
image_size = (128, 128)  # 先对齐尺寸，HOG/GLCM对尺度不敏感但统一更稳
lbp_radius, lbp_points = 1, 8
lbp_method = "uniform"   # 产生 59 种统一模式
glcm_distances = [1, 2, 3]
glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm_props = ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]

# --------------------
# 将 59 类 uniform LBP 压到 15 桶（模拟论文里的 DLBP→GLCM 15×15）
# 固定等宽分桶，确保可复现；需要对齐论文时只需替换这个映射。
# --------------------
def lbp59_to_dlbp15(uniform_lbp_image):
    # uniform LBP 的取值范围：0..58（非uniform并入最后一类）
    bins = np.linspace(-0.5, 58.5, 16)  # 15 段
    dlbp = np.digitize(uniform_lbp_image, bins) - 1  # → 0..14
    return dlbp.astype(np.uint8)

# --------------------
# 单图提取特征：Uniform LBP → 压成 15 类的“DLBP图” → 在其上做 GLCM → 汇聚 Haralick 特征
# --------------------
def extract_dlbp_glcm_features(gray):
    g = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)

    # 1) Uniform LBP（59类）
    ulbp = local_binary_pattern(g, lbp_points, lbp_radius, method=lbp_method)

    # 2) 压成 15 类（DLBP）
    dlbp = lbp59_to_dlbp15(ulbp)

    # 3) 在 DLBP 图上计算 GLCM（levels=15）
    glcm = graycomatrix(dlbp, 
                        distances=glcm_distances, 
                        angles=glcm_angles, 
                        levels=15, 
                        symmetric=True, 
                        normed=True)

    # 4) 汇聚多距离/角度的统计量：取均值与标准差
    feats = []
    for p in glcm_props:
        vals = graycoprops(glcm, p).ravel()     # shape = len(distances)*len(angles)
        feats.append(vals.mean())
        feats.append(vals.std())
    return np.array(feats, dtype=np.float32)    # 最终特征维度 = len(props)*2

# --------------------
# 数据加载
# --------------------
def load_split(split_dir):
    X, y, classes = [], [], []
    if not os.path.isdir(split_dir):
        return None, None, None
    classes = sorted([d for d in os.listdir(split_dir) 
                      if os.path.isdir(os.path.join(split_dir, d))])
    for label_idx, cls in enumerate(classes):
        p_cls = os.path.join(split_dir, cls)
        for fname in os.listdir(p_cls):
            fpath = os.path.join(p_cls, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            feat = extract_dlbp_glcm_features(img)
            X.append(feat); y.append(label_idx)
    if len(X)==0:
        return np.empty((0,)), np.empty((0,)), classes
    return np.stack(X, axis=0), np.array(y, dtype=np.int32), classes

def ensure_train_val(train_dir, val_dir):
    Xtr, ytr, classes = load_split(train_dir)
    if Xtr is None or Xtr.size == 0:
        raise RuntimeError(f"训练集为空：{train_dir}")
    # 如果没有 val，就从 train 按 8:2 切
    Xva, yva, _ = load_split(val_dir)
    if Xva is None or Xva.size == 0:
        n = Xtr.shape[0]
        idx = np.random.RandomState(42).permutation(n)
        cut = int(n*0.8)
        tr_idx, va_idx = idx[:cut], idx[cut:]
        return (Xtr[tr_idx], ytr[tr_idx]), (Xtr[va_idx], ytr[va_idx]), classes
    return (Xtr, ytr), (Xva, yva), classes

# --------------------
# 主流程（纯 GPU 分类，CPU 只做特征）
# --------------------
def main():
    (Xtr, ytr), (Xva, yva), classes = ensure_train_val(train_dir, val_dir)
    print(f"Train: {Xtr.shape}, Val: {Xva.shape}, num_classes={len(classes)}, classes={classes}")

    # 送到 GPU（cuML 直接吃 CuPy）
    Xtr_gpu, ytr_gpu = cp.asarray(Xtr), cp.asarray(ytr, dtype=cp.int32)
    Xva_gpu, yva_gpu = cp.asarray(Xva), cp.asarray(yva, dtype=cp.int32)

    # 使用 GPU 加速训练
    clf = LinearSVC(C=1.0, max_iter=5000)
    clf.fit(Xtr_gpu, ytr_gpu)

    # 训练集预测
    y_train_pred = clf.predict(Xtr_gpu)
    train_acc = accuracy_score(ytr_gpu, y_train_pred)
    print(f"训练集准确率: {train_acc:.4f}")

    # 验证
    y_pred = clf.predict(Xva_gpu)
    val_acc = accuracy_score(yva_gpu, y_pred)
    print(f"验证集准确率: {val_acc:.4f}")

    # 保存模型
    joblib.dump(clf, "lbp_orb_linearSVC_model.pkl")
    print("模型已保存: lbp_orb_linearSVC_model.pkl")

    # 保存日志
    df_log = pd.DataFrame({
        "Train Accuracy": [train_acc],
        "Validation Accuracy": [val_acc]
    })
    df_log.to_excel("training_log.xlsx", index=False)
    print("日志已保存: training_log.xlsx")

    # 画图
    plt.figure()
    plt.bar(["Train", "Validation"], [train_acc, val_acc])
    plt.ylabel("Accuracy")
    plt.title("LBP+ORB Facial Expression Recognition")
    plt.savefig("accuracy_plot.png")
    print("准确率图已保存: accuracy_plot.png")


if __name__ == "__main__":
    main()
