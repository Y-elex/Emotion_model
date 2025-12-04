# train.py
import os, joblib, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import FeatureDataset
from model import FuseMLP
from utils import save_log
from config import *
from tqdm import tqdm

data = 'RAF-DB'
def main():
    device = DEVICE
    # load features
    tr = np.load(os.path.join(FEAT_DIR, f"train_bow_cnn_{data}.npz"))
    val = np.load(os.path.join(FEAT_DIR, f"val_bow_cnn_{data}.npz"))
    Xtr_bow, Xtr_cnn, ytr = tr['bow'], tr['cnn'], tr['y']
    Xv_bow, Xv_cnn, yv = val['bow'], val['cnn'], val['y']

    # scale features (fit on train)
    bow_scaler = StandardScaler()
    cnn_scaler = StandardScaler()
    Xtr_bow = bow_scaler.fit_transform(Xtr_bow)
    Xtr_cnn = cnn_scaler.fit_transform(Xtr_cnn)
    Xv_bow = bow_scaler.transform(Xv_bow)
    Xv_cnn = cnn_scaler.transform(Xv_cnn)

    joblib.dump((bow_scaler, cnn_scaler), os.path.join(OUT_DIR, f"scalers_{data}.pkl"))

    # save scaled npz for dataset
    np.savez(os.path.join(FEAT_DIR, f"train_scaled_{data}.npz"), bow=Xtr_bow, cnn=Xtr_cnn, y=ytr)
    np.savez(os.path.join(FEAT_DIR, f"val_scaled_{data}.npz"), bow=Xv_bow, cnn=Xv_cnn, y=yv)

    train_ds = FeatureDataset(os.path.join(FEAT_DIR, f"train_scaled_{data}.npz"))
    val_ds = FeatureDataset(os.path.join(FEAT_DIR, f"val_scaled_{data}.npz"))

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    model = FuseMLP(bow_dim=Xtr_bow.shape[1], cnn_dim=Xtr_cnn.shape[1], num_classes=len(np.unique(ytr))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    history = {"epoch":[],"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for bow, cnn, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            bow = bow.to(device); cnn = cnn.to(device); labels = labels.to(device)
            outputs = model(bow, cnn)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # val
        model.eval()
        v_loss = 0.0; v_correct=0; v_total=0
        with torch.no_grad():
            for bow, cnn, labels in val_loader:
                bow = bow.to(device); cnn = cnn.to(device); labels = labels.to(device)
                outputs = model(bow, cnn)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_loss = v_loss / v_total
        val_acc = 100.0 * v_correct / v_total

        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} Train Loss {train_loss:.4f} Train Acc {train_acc:.2f}% | Val Loss {val_loss:.4f} Val Acc {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_acc
            }, MODEL_PATH)
            print("Saved best model:", MODEL_PATH)

    save_log(history, LOG_XLSX)
    print("Saved training log to", LOG_XLSX)

if __name__ == "__main__":
    main()
