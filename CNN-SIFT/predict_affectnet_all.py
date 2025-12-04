import os
import argparse
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


def load_and_scale_split(features_dir: str, split: str, dataset_name: str, bow_scaler, cnn_scaler):
    npz_path = os.path.join(features_dir, f"{split}_bow_cnn_{dataset_name}.npz")
    data = np.load(npz_path, allow_pickle=True)

    Xt_bow = data["bow"]
    Xt_cnn = data["cnn"]
    yt = data["y"]

    if "paths" in data:
        paths = data["paths"]
    else:
        paths = np.array([f"{split}_sample_{i}" for i in range(len(yt))])

    Xt_bow = bow_scaler.transform(Xt_bow)
    Xt_cnn = cnn_scaler.transform(Xt_cnn)

    bow_tensor = torch.from_numpy(Xt_bow.astype("float32"))
    cnn_tensor = torch.from_numpy(Xt_cnn.astype("float32"))
    y_tensor = torch.from_numpy(yt.astype("int64"))

    dataset = TensorDataset(bow_tensor, cnn_tensor, y_tensor)
    return dataset, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="AffectNet", choices=["AffectNet", "FER-2013", "FERPlus", "RAF-DB"], help="Dataset name to predict")
    args = parser.parse_args()
    dataset_name = args.dataset

    # Resolve paths relative to this file to avoid CWD issues
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    features_dir = os.path.join(out_dir, "feats")
    model_path = os.path.join(out_dir, f"best_model_{dataset_name}.pth")
    scalers_path = os.path.join(out_dir, f"scalers_{dataset_name}.pkl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    # Load scalers
    bow_scaler, cnn_scaler = joblib.load(scalers_path)

    # Load one split to infer dims for model definition
    probe = np.load(os.path.join(features_dir, f"train_bow_cnn_{dataset_name}.npz"))
    bow_dim = probe["bow"].shape[1]
    cnn_dim = probe["cnn"].shape[1]
    num_classes = len(np.unique(probe["y"]))

    # Import model locally to avoid circular config usage
    from model import FuseMLP

    model = FuseMLP(bow_dim=bow_dim, cnn_dim=cnn_dim, num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_rows = []
    splits = ["train", "val", "test"]

    with torch.no_grad():
        for split in splits:
            dataset, paths = load_and_scale_split(features_dir, split, dataset_name, bow_scaler, cnn_scaler)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            idx_offset = 0
            for bow_batch, cnn_batch, y_batch in loader:
                bow_batch = bow_batch.to(device)
                cnn_batch = cnn_batch.to(device)

                logits = model(bow_batch, cnn_batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                y_true = y_batch.cpu().numpy()

                batch_size_actual = y_true.shape[0]
                batch_paths = paths[idx_offset: idx_offset + batch_size_actual]
                idx_offset += batch_size_actual

                for p, t, pr in zip(batch_paths, y_true, preds):
                    all_rows.append([split, p, int(t), int(pr)])

    df = pd.DataFrame(all_rows, columns=["split", "image_path", "true_label", "pred_label"])
    save_path = os.path.join(out_dir, f"predict_results_{dataset_name}_all.xlsx")
    df.to_excel(save_path, index=False)
    print(f"Saved predictions to {save_path}")


if __name__ == "__main__":
    main()


