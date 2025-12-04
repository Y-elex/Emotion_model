# feature_extract.py
import os, cv2, math, joblib, argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import torch
import torchvision.transforms as T
import torchvision.models as models

# ---------- config ----------
IMG_SIZE = 128         # resize side
SIFT_STEP = 8          # dense grid step
SIFT_PATCH = 16        # keypoint size
BOW_K = 512            # visual words
SAMPLE_DESC = 200000   # descriptors sampled to fit kmeans
BATCH = 64
# ---------------------------

def dense_sift_descriptors(img, step=SIFT_STEP, patch=SIFT_PATCH):
    h, w = img.shape[:2]
    kps = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            kps.append(cv2.KeyPoint(x + patch/2.0, y + patch/2.0, patch))
    # create SIFT
    try:
        sift = cv2.SIFT_create()
    except:
        sift = cv2.xfeatures2d.SIFT_create()
    _, desc = sift.compute(img, kps)
    return desc  # (N,128) or None

def load_images_list(root):
    items = []
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir): continue
        for fn in os.listdir(cls_dir):
            items.append((os.path.join(cls_dir, fn), int(cls)))
    return items

def build_bow_vocab(train_items, out_k=BOW_K, sample_desc=SAMPLE_DESC):
    # sample descriptors randomly from train set
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
    # random subsample to fit kmeans
    if descs.shape[0] > sample_desc:
        idx = np.random.choice(descs.shape[0], sample_desc, replace=False)
        descs = descs[idx]
    print("Fitting kmeans on descriptors:", descs.shape)
    kmeans = MiniBatchKMeans(n_clusters=out_k, batch_size=4096, verbose=1)
    kmeans.fit(descs)
    return kmeans

def build_bow_hist(desc, kmeans):
    if desc is None:
        return np.zeros(kmeans.n_clusters, dtype=np.float32)
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters+1))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

@torch.no_grad()
def extract_cnn_feats(model, imgs, device):
    # imgs: list of single-channel grayscale np arrays resized to IMG_SIZE
    # convert to 3-channel, normalize with ImageNet stats
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    batch_t = torch.stack([transform(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)) for im in imgs]).to(device)
    feats = model(batch_t).cpu().numpy()
    return feats

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) prepare lists
    train_items = load_images_list(os.path.join(args.data, 'train'))
    val_items = load_images_list(os.path.join(args.data, 'val'))
    test_items = load_images_list(os.path.join(args.data, 'test'))

    # 2) build or load kmeans (vocab)
    if args.kmeans and os.path.exists(args.kmeans):
        kmeans = joblib.load(args.kmeans)
        print("Loaded kmeans:", args.kmeans)
    else:
        kmeans = build_bow_vocab(train_items, out_k=args.k, sample_desc=args.sample_desc)
        joblib.dump(kmeans, "kmeans_bow.pkl")
        print("Saved kmeans_bow.pkl")

    # 3) prepare CNN backbone (ResNet50 w/o fc)
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # output (B,2048,1,1)
    resnet.eval().to(device)

    def process_and_save(items, split_name):
        feats_bow = []
        feats_cnn = []
        labels = []
        imgs_batch = []
        paths_batch = []
        for (p, lbl) in tqdm(items, desc=f"Processing {split_name}"):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            desc = dense_sift_descriptors(img)
            hist = build_bow_hist(desc, kmeans)
            feats_bow.append(hist)
            imgs_batch.append(img)
            paths_batch.append(p)
            labels.append(lbl)
            if len(imgs_batch) >= args.batch:
                # CNN feats
                cnn_feats = extract_cnn_feats(resnet, imgs_batch, device)  # shape (B,2048,1,1)
                cnn_feats = cnn_feats.reshape(cnn_feats.shape[0], -1)
                feats_cnn.append(cnn_feats)
                imgs_batch = []
        # leftover
        if imgs_batch:
            cnn_feats = extract_cnn_feats(resnet, imgs_batch, device)
            feats_cnn.append(cnn_feats)
        feats_cnn = np.vstack(feats_cnn) if feats_cnn else np.zeros((len(feats_bow), 2048))
        feats_bow = np.vstack(feats_bow)
        labels = np.array(labels)
        # save
        np.savez(f"{split_name}_bow_cnn.npz", bow=feats_bow, cnn=feats_cnn, y=labels)
        print(f"Saved {split_name}_bow_cnn.npz: bow {feats_bow.shape}, cnn {feats_cnn.shape}")
        return

    process_and_save(train_items, "train")
    process_and_save(val_items, "val")
    process_and_save(test_items, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../datasets/AffectNet', help='dataset root')
    parser.add_argument('--k', type=int, default=BOW_K)
    parser.add_argument('--kmeans', default=None, help='path to kmeans.pkl to load')
    parser.add_argument('--sample_desc', type=int, default=SAMPLE_DESC)
    parser.add_argument('--batch', type=int, default=BATCH)
    args = parser.parse_args()
    main(args)
