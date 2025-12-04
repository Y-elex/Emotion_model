import os
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from dataset import FeatureDataset
from model import FuseMLP
from config import *

data = "FERPlus"

def main():
    device = DEVICE

    # === 1. 加载 scaler ===
    bow_scaler, cnn_scaler = joblib.load(os.path.join(OUT_DIR, f"scalers_{data}.pkl"))

    # === 2. 加载 val 特征 ===
    test = np.load(os.path.join(FEAT_DIR, f"val_bow_cnn_{data}.npz"), allow_pickle=True)
    Xt_bow, Xt_cnn, yt = test["bow"], test["cnn"], test["y"]

    # 如果有路径就用路径，否则用索引号代替
    if "paths" in test:
        paths = test["paths"]
    else:
        paths = np.array([f"sample_{i}" for i in range(len(yt))])

    # === 3. scale val ===
    Xt_bow = bow_scaler.transform(Xt_bow)
    Xt_cnn = cnn_scaler.transform(Xt_cnn)

    # 保存 scaled npz（带路径）
    np.savez(os.path.join(FEAT_DIR, f"val_scaled_{data}.npz"),
             bow=Xt_bow, cnn=Xt_cnn, y=yt, paths=paths)

    test_ds = FeatureDataset(os.path.join(FEAT_DIR, f"val_scaled_{data}.npz"))
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    # === 4. 初始化模型 ===
    model = FuseMLP(
        bow_dim=Xt_bow.shape[1],
        cnn_dim=Xt_cnn.shape[1],
        num_classes=len(np.unique(yt))
    ).to(device)

    # === 5. 加载权重 ===
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # === 6. 预测 ===
    results = []
    with torch.no_grad():
        idx = 0
        for bow, cnn, labels in test_loader:
            bow, cnn = bow.to(device), cnn.to(device)
            outputs = model(bow, cnn)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            batch_size = len(labels)
            batch_paths = paths[idx: idx+batch_size]
            idx += batch_size

            for p, true, pred in zip(batch_paths, labels, preds):
                results.append(["test", p, true, pred])

    # === 7. 保存结果 ===
    df = pd.DataFrame(results, columns=["split", "image_path", "true_label", "pred_label"])
    out_path = os.path.join(OUT_DIR, f"predict_results_{data}_val.xlsx")
    df.to_excel(out_path, index=False)
    print("✅ 预测完成，结果已保存到", out_path)

if __name__ == "__main__":
    main()
