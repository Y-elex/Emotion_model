# train_fuse.py
import numpy as np, joblib, argparse
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def main(args):
    # load features
    tr = np.load("train_bow_cnn.npz")
    val = np.load("val_bow_cnn.npz")
    X_tr = np.hstack([tr['bow'], tr['cnn']])
    y_tr = tr['y']
    X_val = np.hstack([val['bow'], val['cnn']])
    y_val = val['y']

    # scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # classifier (MLP)
    clf = MLPClassifier(hidden_layer_sizes=(1024,256), activation='relu', batch_size=128, max_iter=50, verbose=True)
    clf.fit(X_tr, y_tr)

    # eval
    ytr_pred = clf.predict(X_tr)
    yval_pred = clf.predict(X_val)
    print("Train acc:", accuracy_score(y_tr, ytr_pred))
    print("Val acc:", accuracy_score(y_val, yval_pred))
    print(classification_report(y_val, yval_pred))

    # save
    joblib.dump(clf, "fuse_mlp.pkl")
    joblib.dump(scaler, "scaler_fuse.pkl")
    print("Saved classifier & scaler")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
