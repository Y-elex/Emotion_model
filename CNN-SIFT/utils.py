# utils.py
import pandas as pd
import os

def save_log(history, path):
    """
    history: dict of lists, keys: epoch, train_loss, train_acc, val_loss, val_acc
    """
    df = pd.DataFrame(history)
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    df.to_excel(path, index=False)
