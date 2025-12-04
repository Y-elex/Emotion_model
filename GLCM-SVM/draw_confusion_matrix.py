import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 读取预测结果 Excel 文件
data = 'AffectNet'  # 替换为你的数据集名称
excel_file = f'E:/python code/results/prediction/GLCM/prediction_{data}_val.xlsx'  # 修改为你的文件名
df = pd.read_excel(excel_file)

# 提取真实标签和预测标签
y_true = df['true_label']
y_pred = df['pred_label']

# 获取类别名（假设是 AffectNet）
label_map = {
    0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise',
    4: 'Fear', 5: 'Disgust', 6: 'Anger', 7: 'Contempt'
}
labels = list(label_map.keys())
label_names = [label_map[i] for i in labels]

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

# 设置图像尺寸和样式
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)

# 绘制热力图
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.title('Confusion Matrix')

# 保存为 PNG
plt.tight_layout()
plt.savefig(f'E:/python code/results/figures/GLCM/confusion_matrix_{data}_val.png', dpi=300)


print(f"✅ 混淆矩阵图已保存为 confusion_matrix_{data}_val.png")
