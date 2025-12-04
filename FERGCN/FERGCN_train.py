import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from dataset import LandmarkGraphDataset
from model import FERGCN  # 你的 GCN 模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = 'RAF-DB'

# 参数
batch_size = 32
num_epochs = 100
learning_rate = 1e-3

# 数据
train_dataset = LandmarkGraphDataset(f"./landmarks_{data}/train")
val_dataset = LandmarkGraphDataset(f"./landmarks_{data}/val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 模型
model = FERGCN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 日志
log_path = f"train_log_{data}.xlsx"
log_df = pd.DataFrame(columns=["epoch","train_loss","train_acc","val_loss","val_acc"])

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
    for nodes, adjs, labels in loop:
        nodes, adjs, labels = nodes.to(device), adjs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(nodes, adjs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    train_loss = running_loss / total
    train_acc = correct / total

    # 验证
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for nodes, adjs, labels in val_loader:
            nodes, adjs, labels = nodes.to(device), adjs.to(device), labels.to(device)
            outputs = model(nodes, adjs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            _, pred = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 保存日志
    log_df.loc[epoch] = [epoch+1, train_loss, train_acc, val_loss, val_acc]
    log_df.to_excel(log_path, index=False)

# 保存模型
torch.save(model.state_dict(), f"fergcn_model_{data}.pth")
