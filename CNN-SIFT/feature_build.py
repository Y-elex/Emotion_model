# feature_build.py
import os, cv2, joblib, argparse, numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import torch, torchvision.models as models, torchvision.transforms as T
from config import *

data = 'RAF-DB'
def dense_sift_descriptors(img, step=SIFT_STEP, patch=SIFT_PATCH):
    h, w = img.shape[:2]
    kps = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            kps.append(cv2.KeyPoint(float(x + patch/2.0), float(y + patch/2.0), float(patch)))
    try:
        sift = cv2.SIFT_create()
    except:
        sift = cv2.xfeatures2d.SIFT_create()
    _, desc = sift.compute(img, kps)
    return desc

def load_items(split):
    items = []
    root = os.path.join(DATA_ROOT, split)
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir): continue
        for fn in os.listdir(cls_dir):
            items.append((os.path.join(cls_dir, fn), int(cls)))
    return items

@torch.no_grad()
def extract_cnn_feats(model, imgs, device):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    batch_t = torch.stack([transform(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)) for im in imgs]).to(device)
    feats = model(batch_t).cpu().numpy().reshape(len(imgs), -1)
    return feats

def build_kmeans(train_items, sample_desc=SAMPLE_DESC, k=BOW_K):
    descs = []
    cnt = 0
    for p, _ in tqdm(train_items, desc="sampling descriptors"):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        d = dense_sift_descriptors(img)
        if d is None: continue
        descs.append(d)
        cnt += d.shape[0]
        if cnt > sample_desc:
            break
    descs = np.vstack(descs)
    if descs.shape[0] > sample_desc:
        idx = np.random.choice(descs.shape[0], sample_desc, replace=False)
        descs = descs[idx]
    print("KMeans fit on descriptors:", descs.shape)
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, verbose=1)
    kmeans.fit(descs)
    joblib.dump(kmeans, KMEANS_PATH)
    return kmeans

def build_bow_hist(desc, kmeans):
    if desc is None:
        return np.zeros(kmeans.n_clusters, dtype=np.float32)
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters+1))
    hist = hist.astype(np.float32)
    if hist.sum()>0:
        hist /= hist.sum()
    return hist

def process_split(items, kmeans, resnet, device, out_file):
    bows, cnns, labels = [], [], []
    imgs_batch, idx_batch = [], []
    for (p, lbl) in tqdm(items, desc=f"Process {out_file}"):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        desc = dense_sift_descriptors(img)
        bows.append(build_bow_hist(desc, kmeans))
        imgs_batch.append(img)
        labels.append(lbl)
        # batch process CNN
        if len(imgs_batch) >= 64:
            feats = extract_cnn_feats(resnet, imgs_batch, device)
            cnns.append(feats)
            imgs_batch=[]
    if imgs_batch:
        feats = extract_cnn_feats(resnet, imgs_batch, device)
        cnns.append(feats)
    cnns = np.vstack(cnns) if len(cnns)>0 else np.zeros((len(bows),2048))
    bows = np.vstack(bows)
    labels = np.array(labels)
    np.savez(os.path.join(FEAT_DIR, out_file), bow=bows, cnn=cnns, y=labels)
    print("Saved", out_file)

def main():
    device = DEVICE
    train_items = load_items("train")
    val_items = load_items("val")
    test_items = load_items("test")

    if os.path.exists(KMEANS_PATH):
        kmeans = joblib.load(KMEANS_PATH)
        print("Loaded existing kmeans.")
    else:
        kmeans = build_kmeans(train_items)

    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet.eval()

    process_split(train_items, kmeans, resnet, device, f"train_bow_cnn_{data}.npz")
    process_split(val_items, kmeans, resnet, device, f"val_bow_cnn_{data}.npz")
    process_split(test_items, kmeans, resnet, device, f"test_bow_cnn_{data}.npz")

if __name__ == "__main__":
    main()
