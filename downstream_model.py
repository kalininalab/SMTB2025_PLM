from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import copy
import pandas as pd
import numpy as np
import torch
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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True) 
parser.add_argument('--layer', type=int, required=True)
parser.add_argument('--function', type=str, required=True)
args = parser.parse_args()


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
    targets = torch.tensor(df['label'].values, dtype=torch.float)
    inputs = inputs.numpy()
    targets = targets.numpy()
    print("dataset created")
    return inputs, targets


train_X, train_Y = build_dataloader(df[df["split"] == "train"], f"/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{args.layer}/")
valid_X, valid_Y = build_dataloader(df[df["split"] == "valid"], f"/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{args.layer}/")
test_X, test_Y = build_dataloader(df[df["split"] == "test"], f"/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{args.layer}/")

if {args.function} == "linereg":
    reg = LinearRegression().fit(train_X, train_Y)
elif {args.function} == "randomforest":
    reg = RandomForestRegressor().fit(train_X, train_Y)

train_prediction = reg.predict(train_X)
test_prediction = reg.predict(test_X)
valid_prediction = reg.predict(valid_X)

train_spearman = spearmanr(train_prediction, train_Y)[0]
valid_spearman = spearmanr(valid_prediction, valid_Y)[0]
test_spearman = spearmanr(test_prediction, test_Y)[0]

pd.DataFrame({
    "train_spearman": [train_spearman],
    "valid_spearman": [valid_spearman],
    "test_spearman": [test_spearman],
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
        f.write("Embedding Model, Downstream Model, #layers, Dataset, Spearman\n")  # Create a table

with open("/srv/scratch/PLM/results.csv", "a") as f:
    f.write(f"{args.model}, {args.fuction}, {args.layer}, {args.dataset}, {test_spearman}\n")

