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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--esm_model', type=str, required=True) 
parser.add_argument('--layer', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
args = parser.parse_args()

#extracting the number of model
n = args.esm_model[6:8]
if "_" in n:
    n = args.esm_model[6:7]
n = int(n)

name = args.dataset[:-4]

df = pd.read_csv(f"/srv/scratch/PLM/datasets/{args.dataset}")

torch.manual_seed(42)

def build_dataloader(df: pd.DataFrame, embed_path: Path, **dataloader_kwargs: Any) -> DataLoader:
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
    return DataLoader(TensorDataset(inputs, targets), **dataloader_kwargs)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(x)
    

#Create a model

TRAIN_EPOCHES = 50

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)
loss = nn.MSELoss()

best_val_loss = float('inf')
best_model = model

train = build_dataloader(df[df["split"] == "train"], f"/srv/scratch/PLM/embeddings/esm_t{n}/{name}/layer_{args.layer}/", batch_size=args.batch_size, shuffle=True, pin_memory=True)
valid = build_dataloader(df[df["split"] == "valid"], f"/srv/scratch/PLM/embeddings/esm_t{n}/{name}/layer_{args.layer}/", batch_size=args.batch_size, shuffle=False, pin_memory=True)
test = build_dataloader(df[df["split"] == "test"], f"/srv/scratch/PLM/embeddings/esm_t{n}/{name}/layer_{args.layer}/", batch_size=args.batch_size, shuffle=False, pin_memory=True)

train_loss = []
train_spearman = []
train_pearson = []
valid_loss = []
valid_spearman = []
valid_pearson = []

for e in tqdm(range(TRAIN_EPOCHES)):
    model.train()
    train_errors = []
    train_ys = []
    train_y_hats = []
    for batch in train:
        train_x, train_y = batch
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        
        optimizer.zero_grad()
        train_y_hat = model(train_x)
        train_error = loss(train_y_hat, train_y)
        train_error.backward()
        optimizer.step()
        
        train_errors.append(train_error.detach().cpu().numpy())
        train_ys += list(train_y.detach().cpu().numpy().flatten())
        train_y_hats += list(train_y_hat.detach().cpu().numpy().flatten())
    
    train_loss.append(np.mean(train_errors))
    train_spearman.append(spearmanr(train_y_hats, train_ys)[0])
    train_pearson.append(np.corrcoef(train_y_hats, train_ys)[0, 1])

    with torch.no_grad():
        model.eval()
        v_errors = []
        valid_ys = []
        valid_y_hats = []
        for batch in valid:
            valid_x, valid_y = batch
            valid_x = valid_x.to(device)
            valid_y = valid_y.to(device)
            
            val_y_hat = model(valid_x)
            valid_error = loss(val_y_hat, valid_y)

            v_errors.append(valid_error.detach().cpu().numpy())
            valid_ys += list(valid_y.detach().cpu().numpy().flatten())
            valid_y_hats += list(val_y_hat.detach().cpu().numpy().flatten())

        val_loss = np.mean(v_errors)
        valid_loss.append(val_loss)
        valid_spearman.append(spearmanr(valid_y_hats, valid_ys)[0])
        valid_pearson.append(np.corrcoef(valid_y_hats, valid_ys)[0, 1])
        scheduler.step(valid_loss[-1])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_impromevent = 0
            best_model = copy.deepcopy(model)

with torch.no_grad():
    best_model.eval()
    t_errors = []
    test_y_hats = []
    test_ys = []
    for batch in test:
        test_x, test_y = batch
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        
        test_y_hat = best_model(test_x)
        test_error = loss(test_y_hat, test_y)

        test_ys += list(test_y.detach().cpu().numpy().flatten())
        test_y_hats += list(test_y_hat.detach().cpu().numpy().flatten())
        t_errors.append(test_error.detach().cpu().numpy())

perf = np.mean(t_errors)
spearman = spearmanr(test_y_hats, test_ys)[0]
pearson = np.corrcoef(test_y_hats, test_ys)[0, 1]

pd.DataFrame({
    "train_loss": train_loss, 
    "train_spear": train_spearman, 
    "train_pearson": train_pearson,
    "val_loss": valid_loss, 
    "valid_spearman": valid_spearman,
    "valid_pearson": valid_pearson,
}).to_csv(f"/srv/scratch/PLM/embeddings/esm_t{n}/{name}/layer_{args.layer}/losses_h.csv", index=False)

df_predictions = pd.DataFrame({"y": test_ys,"y_hats": test_y_hats})
df_predictions.to_csv(f"/srv/scratch/PLM/embeddings/esm_t{n}/{name}/layer_{args.layer}/predictions_h.csv", index=False)

if not os.path.exists("/srv/scratch/PLM/results_h.csv"):
    with open("/srv/scratch/PLM/results_h.csv", "w") as f:
        f.write("Model Name, #layers, Dataset, Performance, Spearman, Pearson\n")  # Create a table

with open("/srv/scratch/PLM/results_h.csv", "a") as f:
    f.write(f"{args.esm_model}, {args.layer}, {args.dataset}, {perf}, {spearman}, {pearson}\n")
