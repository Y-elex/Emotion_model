# config.py
import os

data = 'RAF-DB'  # 可选 'FERPlus', 'AffectNet', 'RAF-DB'
DATA_ROOT = f"../datasets/{data}"   # 你的数据集根目录
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
KMEANS_PATH = os.path.join(OUT_DIR, f"kmeans_bow_{data}.pkl")
FEAT_DIR = os.path.join(OUT_DIR, "feats")   # 保存 train_bow_cnn.npz 等
os.makedirs(FEAT_DIR, exist_ok=True)

IMG_SIZE = 128      # resize for SIFT & CNN
BOW_K = 512         # visual words
SAMPLE_DESC = 200000
SIFT_STEP = 8
SIFT_PATCH = 16

BATCH = 64
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

LOG_XLSX = os.path.join(OUT_DIR, f"train_log_{data}.xlsx")
MODEL_PATH = os.path.join(OUT_DIR, f"best_model_{data}.pth")
SAVED_SCALER = os.path.join(OUT_DIR, f"scaler_fuse_{data}.pkl")
