import os
import cv2
import cupy as cp
import numpy as np
import joblib
import pandas as pd

# ===================== 参数 =====================
data = "AffectNet"
data_dir = f"E:/python code/FER/datasets/{data}"
image_size = (128, 128)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# HOG 描述子
hog = cv2.HOGDescriptor(
    _winSize=(image_size[1] // pixels_per_cell[1] * pixels_per_cell[1],
              image_size[0] // pixels_per_cell[0] * pixels_per_cell[0]),
    _blockSize=(cells_per_block[1] * pixels_per_cell[1],
                cells_per_block[0] * pixels_per_cell[0]),
    _blockStride=(pixels_per_cell[1], pixels_per_cell[0]),
    _cellSize=(pixels_per_cell[1], pixels_per_cell[0]),
    _nbins=orientations
)

def extract_hog(image_gray):
    img = cv2.resize(image_gray, image_size)
    return hog.compute(img).flatten()

# ===================== 预测单个文件夹 =====================
def predict_folder(folder_path, models, classes):
    img_paths, true_labels, pred_labels = [], [], []
    true_label_ids, pred_label_ids = [], []
    feats = []

    for label_idx, cls in enumerate(classes):  # 按训练的类别顺序
        cls_dir = os.path.join(folder_path, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = extract_hog(img)
            img_paths.append(fpath)
            true_labels.append(cls)      # 真实类别名
            true_label_ids.append(label_idx)  # 真实类别ID
            feats.append(feat)

    if len(feats) == 0:
        return [], [], [], [], []

    X_gpu = cp.asarray(np.stack(feats, axis=0), dtype=cp.float32)

    # OVR 多分类预测
    scores = []
    for clf in models:
        scores.append(clf.predict(X_gpu).astype(cp.float32))
    scores = cp.stack(scores, axis=1)
    y_pred_idx = cp.argmax(scores, axis=1).get()
    y_pred_labels = [classes[idx] for idx in y_pred_idx]
    pred_label_ids = y_pred_idx.tolist()

    return img_paths, true_labels, true_label_ids, y_pred_labels, pred_label_ids

# ===================== 主流程 =====================
def main():
    # 加载训练好的模型
    model_path = f"E:/python code/results/models/HOGhog_svm_gpu_ovr_{data}_models.pkl"
    models = joblib.load(model_path)
    print(f"模型已加载: {model_path}")

    # 根据训练集文件夹确定类别顺序
    train_dir = os.path.join(data_dir, "train")
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"类别列表: {classes}")

    all_splits = ['train', 'val', 'test']
    df_all = pd.DataFrame(columns=['Split', 'Image', 'True_Label', 'True_Label_Id', 'Pred_Label', 'Pred_Label_Id'])

    for split in all_splits:
        folder = os.path.join(data_dir, split)
        if not os.path.exists(folder):
            print(f"跳过不存在的文件夹: {folder}")
            continue
        print(f"预测 {split} 文件夹...")
        img_paths, true_labels, true_label_ids, pred_labels, pred_label_ids = predict_folder(folder, models, classes)
        df_split = pd.DataFrame({
            'Split': [split] * len(img_paths),
            'Image': img_paths,
            'True_Label': true_labels,
            'True_Label_Id': true_label_ids,
            'Pred_Label': pred_labels,
            'Pred_Label_Id': pred_label_ids
        })
        df_all = pd.concat([df_all, df_split], ignore_index=True)

    # 保存 Excel
    output_file = f"predictions_hog_gpu_{data}.xlsx"
    df_all.to_excel(output_file, index=False)
    print(f"预测结果已保存: {output_file}")

if __name__ == "__main__":
    main()
