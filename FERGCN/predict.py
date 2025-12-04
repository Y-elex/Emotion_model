import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from collections import defaultdict
from dataset import LandmarkGraphDataset
from model import FERGCN  # 你的GCN模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "FER-2013"

# === 配置 ===
batch_size = 32
num_classes = 8  # 根据你的数据集调整：RAF-DB 通常是 8 类

# RAF-DB 的标准 7 类情绪标签（请根据你实际的 label 映射调整！）
label_map = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'  # 如果有第8类，请取消注释
}
# ==================

# 模型路径
model_path = f"fergcn_model_{data}.pth"

# === 新增：检查模型文件是否存在 ===
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型文件不存在: {os.path.abspath(model_path)}\n请先训练模型并保存。")
else:
    print(f"✅ 加载模型: {model_path} ({os.path.getsize(model_path) / 1e6:.2f} MB)")
# ===================================

# 加载模型
model = FERGCN(num_classes=num_classes).to(device)  # 确保 FERGCN 支持 num_classes 参数
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def run_prediction(split_name):
    """对 train/val/test 一个 split 进行预测，并返回结果及标签列表"""
    dataset = LandmarkGraphDataset(f"./landmarks_{data}/{split_name}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = []
    true_labels_all = []
    pred_labels_all = []

    with torch.no_grad():
        for nodes, adjs, labels, paths in loader:
            nodes, adjs, labels = nodes.to(device), adjs.to(device), labels.to(device)
            outputs = model(nodes, adjs)
            _, preds = torch.max(outputs, 1)

            # 收集结果
            for p, true, pred in zip(paths, labels.cpu().numpy(), preds.cpu().numpy()):
                results.append([split_name, p, true, pred])
                true_labels_all.append(true)
                pred_labels_all.append(pred)

    return results, true_labels_all, pred_labels_all

# 执行预测
all_results = []
all_true_labels = []
all_pred_labels = []

for split in ["val"]:  # 可扩展为 ["train", "val", "test"]
    print(f"正在预测 {split} ...")
    results, true_labels, pred_labels = run_prediction(split)
    all_results.extend(results)
    all_true_labels.extend(true_labels)
    all_pred_labels.extend(pred_labels)

# === 计算并打印准确率 ===
total_correct = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == p)
total_samples = len(all_true_labels)
overall_acc = total_correct / total_samples if total_samples > 0 else 0.0

print("\n" + "="*60)
print(f"📊 整体准确率 (Overall Accuracy): {overall_acc:.4f} ({total_correct}/{total_samples})")
print("="*60)

# 每类准确率
per_class_correct = defaultdict(int)
per_class_total = defaultdict(int)

for t, p in zip(all_true_labels, all_pred_labels):
    per_class_total[t] += 1
    if t == p:
        per_class_correct[t] += 1

print("\n📈 各类别表情识别准确率:")
print("-" * 60)
for class_id in range(num_classes):
    class_name = label_map.get(class_id, f"Class {class_id}")
    total = per_class_total[class_id]
    correct = per_class_correct[class_id]
    if total > 0:
        acc = correct / total
        print(f"{class_id:>2d} ({class_name:>12}): {acc:.4f} ({correct:>4d}/{total:>4d})")
    else:
        print(f"{class_id:>2d} ({class_name:>12}): N/A (0/0)")
# ==========================

# 保存结果到 Excel
df = pd.DataFrame(all_results, columns=["split", "image_path", "true_label", "pred_label"])
output_file = f"predict_{data}_val.xlsx"
df.to_excel(output_file, index=False)

print("\n" + "="*60)
print(f"✅ 所有预测完成，结果已保存到: {output_file}")
print("="*60)
