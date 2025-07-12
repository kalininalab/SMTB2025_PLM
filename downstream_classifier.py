from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import copy
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Any
from scipy.stats import spearmanr
from ray import tune
import random
from sklearn.metrics import mean_squared_error


def multioutput_mcc(y_true, y_pred):
    """
    Compute the average Matthews Correlation Coefficient (MCC) for a multi-output task.

    Parameters:
    - y_true: np.ndarray of shape (n_samples, n_outputs)
    - y_pred: np.ndarray of shape (n_samples, n_outputs)

    Returns:
    - float: average MCC across outputs
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    
    mccs = []
    for i in range(y_true.shape[1]):
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
        except ValueError:
            # Handle cases where MCC is undefined (e.g., only one class present)
            mcc = 0.0
        mccs.append(mcc)
    
    return np.mean(mccs)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--dataset', type=str, required=True, help="Dataset name, e.g., 'deeploc2_bin_data.csv'.")
parser.add_argument('--model', type=str, required=True) 
parser.add_argument('--layer', type=int, required=True)
parser.add_argument('--function', type=str, required=True)
args = parser.parse_args()

if os.path.exists("/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{args.layer}/predictions_{args.function}.csv"):
    exit(0)

name = args.dataset[:-4]

df = pd.read_csv(f"/srv/scratch/PLM/datasets/{args.dataset}")

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def build_dataloader(df: pd.DataFrame, embed_path: Path):
    """
    Build a DataLoader for the given DataFrame and embedding path.

    :param df: DataFrame containing the data.
    :param embed_path: Path to the directory containing the embeddings.
    :param dataloader_kwargs: Additional arguments for DataLoader.

    :return: DataLoader for the embeddings and targets.
    """
    embed_path = Path(embed_path)
    embeddings = []
    for idx in df["ID"].values:
        with open(embed_path / f"{idx}.pkl", "rb") as f:
            embeddings.append(pickle.load(f).detach().cpu().float())
    # inputs = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    inputs = torch.stack(embeddings)
    if "bin" not in args.dataset:
        targets = torch.tensor(df[["Cytoplasm", "Nucleus", "Extracellular", "Mitochondrion", "Cell membrane", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]].values, dtype=int)
    else:
        targets = torch.tensor(df['label'].values, dtype=int)
    inputs = inputs.numpy()
    targets = targets.numpy()
    print("dataset created")
    return inputs, targets

train_X, train_Y = build_dataloader(df[df["split"] == "train"], f"/srv/scratch/PLM/embeddings/{args.model}/deeploc2_bin_data/layer_{args.layer}/")
valid_X, valid_Y = build_dataloader(df[df["split"] == "valid"], f"/srv/scratch/PLM/embeddings/{args.model}/deeploc2_bin_data/layer_{args.layer}/")
test_X, test_Y = build_dataloader(df[df["split"] == "test"], f"/srv/scratch/PLM/embeddings/{args.model}/deeploc2_bin_data/layer_{args.layer}/")

if "bin" in args.dataset:
    if args.function == "linereg":
        reg = LogisticRegression().fit(train_X, train_Y)
    elif args.function == "randomforest":
        reg = RandomForestClassifier(n_estimators=40, min_samples_leaf=5, max_depth=20, n_jobs=4).fit(train_X, train_Y)
else:
    if args.function == "linereg":
        reg = MultiOutputClassifier(LogisticRegression()).fit(train_X, train_Y)
    elif args.function == "randomforest":
        reg = MultiOutputClassifier(RandomForestClassifier(n_estimators=40, min_samples_leaf=5, max_depth=20, n_jobs=4)).fit(train_X, train_Y)

train_prediction = reg.predict(train_X)
test_prediction = reg.predict(test_X)
valid_prediction = reg.predict(valid_X)

train_acc = accuracy_score(train_prediction, train_Y)
valid_acc = accuracy_score(valid_prediction, valid_Y)
test_acc = accuracy_score(test_prediction, test_Y)

if "bin" in args.dataset:
    train_mcc = matthews_corrcoef(train_prediction, train_Y)
    valid_mcc = matthews_corrcoef(valid_prediction, valid_Y)
    test_mcc = matthews_corrcoef(test_prediction, test_Y)
else:
    train_mcc = multioutput_mcc(train_Y, train_prediction)
    valid_mcc = multioutput_mcc(valid_Y, valid_prediction)
    test_mcc = multioutput_mcc(test_Y, test_prediction)

Path(f"/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{args.layer}").mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "train_acc": [train_acc],
    "valid_acc": [valid_acc],
    "test_acc": [test_acc],
    "train_mcc": [train_mcc],
    "valid_mcc": [valid_mcc],
    "test_mcc": [test_mcc],
}).to_csv(f"/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{args.layer}/metrics_{args.function}.csv", index=False)

max_len = max(len(train_Y), len(valid_Y), len(test_Y))

train_Y = list(train_Y)  # Convert to list
train_prediction = list(train_prediction)  # Convert to list
valid_Y = list(valid_Y)
valid_prediction = list(valid_prediction)
test_Y = list(test_Y)
test_prediction = list(test_prediction)

train_Y += [0] * (max_len - len(train_Y))
train_prediction += [0] * (max_len - len(train_prediction))
valid_Y += [0] * (max_len - len(valid_Y))
valid_prediction += [0] * (max_len - len(valid_prediction))
test_Y += [0] * (max_len - len(test_Y))
test_prediction += [0] * (max_len - len(test_prediction))

pd.DataFrame({
    "train": train_Y,
    "train_prediction": train_prediction,
    "valid": valid_Y,
    "valid_predictions": valid_prediction,
    "test": test_Y,
    "test_predicted": test_prediction
}).to_csv(f"/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{args.layer}/predictions_{args.function}.csv", index=False)

if not os.path.exists(f"/srv/scratch/PLM/results.csv"):
    with open(f"/srv/scratch/PLM/results.csv", "w") as f:
        f.write("Embedding Model, Downstream Model, #layers, Dataset, Spearman, MSE\n")  # Create a table

with open("/srv/scratch/PLM/results.csv", "a") as f:
    f.write(f"{args.model}, {args.function}, {args.layer}, {args.dataset}, {test_acc}, {test_mcc}\n")
