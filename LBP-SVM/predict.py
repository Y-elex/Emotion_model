import os
import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================
# 参数设置
# ========================
data = 'AffectNet'  # 可选 'CK+'、'AffectNet'、'RAF-DB'
dataset_root = f"../datasets/{data}"
splits = ['val']  # 或者只预测某个子集
model_path = f"lbp_orb_linearSVC_{data}_model.pkl"
output_excel = f"predict_{data}_val.xlsx"

radius = 1
n_points = 8 * radius
METHOD = 'uniform'
orb_features_dim = 32

# ========================
# 特征提取函数
# ========================
def extract_lbp(image_gray):
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(image_gray, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def extract_orb(image_gray, n_features=200):
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
# 加载模型
# ========================
clf = joblib.load(model_path)
print(f"已加载模型: {model_path}")

# ========================
# 预测
# ========================
results = []
id_counter = 1

for split in splits:
    split_dir = os.path.join(dataset_root, split)
    if not os.path.exists(split_dir):
        continue
    print(f"🔍 正在处理子集: {split}")

    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        true_label = int(class_name)

        for img_name in tqdm(os.listdir(class_dir), desc=f"{split}/{class_name}"):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_dir, img_name)
            try:
                feat = extract_features(img_path)
                if feat is None:
                    raise ValueError("无法读取图片或特征提取失败")

                pred_label = clf.predict(feat.reshape(1, -1))[0]

                results.append({
                    'id': id_counter,
                    'split': split,
                    'image_name': os.path.join(split, class_name, img_name),
                    'true_label': true_label,
                    'pred_label': pred_label
                })
                id_counter += 1

            except Exception as e:
                print(f"[错误] 处理失败 {img_path}：{e}")

# ========================
# 保存结果
# ========================
df = pd.DataFrame(results)
df.to_excel(output_excel, index=False)
print(f"✅ 全部预测完成，结果已保存至：{output_excel}")
