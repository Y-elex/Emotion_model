import torch
from model import FERGCN
from dataset import AffectNetGraphDataset

data = 'AffectNet'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FERGCN().to(device)
model.load_state_dict(torch.load(f'checkpoints/best_{data}.pth'))
model.eval()

dataset = AffectNetGraphDataset(f'../datasets/{data}/test')
node_features, adj, _ = dataset[0]
node_features = node_features.unsqueeze(0).to(device)
adj = adj.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(node_features, adj)
    pred = outputs.argmax(dim=1)
    print(f'Predicted class: {pred.item()}')
