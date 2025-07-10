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

torch.manual_seed(25)

def get_data(split):
    print("Load", split)
    x = []
    y = []

    for file in tqdm(os.listdir(f"/srv/scratch/PLM/embeddings/esm_t{n}/{name}/layer_{args.layer}/")):
        index = file[:-4]
        df2 = df[df["ID"] == index]
        if df2["split"].values[0] == split:
            value = df2["label"]
            y.append(value.values[0])
            with open(f"/srv/scratch/PLM/embeddings/esm_t{n}/{name}/layer_{args.layer}/{file}", "rb") as f:
                embedding = pickle.load(f)
            # embedding = pd.read_pickle(f"/srv/scratch/PLM/embeddings/esm_t{n}/{name}/layer_{args.layer}/{file}")
            x.append(embedding)

    x = torch.stack(x)
    y = torch.FloatTensor(y)

    dataset = TensorDataset(x,y)
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return data


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(x)
    
#Create a model

TRAIN_EPOCHES = 50
PATIENCE = 8

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
es_losses = []
loss = nn.MSELoss()
best_val_loss = float('inf')
epochs_without_improvement = 0
train, valid, test = get_data("train"), get_data("valid"), get_data("test")
model.to(device)

for e in tqdm(range(TRAIN_EPOCHES)):
    model.train()
    for batch in train:
        train_x, train_y = batch
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        optimizer.zero_grad()
        train_y_hat = model(train_x)
        train_error = loss(train_y_hat, train_y)
        train_error.backward()
        optimizer.step()


    with torch.no_grad():
        model.eval()
        v_errors = []
        for batch in valid:
            valid_x, valid_y = batch
            valid_x = valid_x.to(device)
            valid_y = valid_y.to(device)
            valid_y_hat = model(valid_x)
            valid_error = loss(valid_y_hat, valid_y)
            valid_error = valid_error.to("cpu")
            v_errors.append(valid_error)

        val_loss = np.mean(v_errors)

        if val_loss < best_val_loss:
            best_val_loss = best_val_loss
            epochs_without_impromevent = 0
        else:
            epochs_without_improvement +=1

        if epochs_without_improvement >= PATIENCE:
            break

with torch.no_grad():
    model.eval()
    t_errors = []
    for batch in test:
        test_x, test_y = batch
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        test_y_hat = model(test_x)
        test_error = loss(test_y_hat, test_y)
        test_error = test_error.to("cpu")
        t_errors.append(test_error)

    test_loss = np.mean(t_errors)


#Save a file with results

if not os.path.exists("/srv/scratch/PLM/results.csv"):
    with open("/srv/scratch/PLM/results.csv", "w") as f:
        f.write("Model Name, #layers, Dataset, Performance\n")  # Create a table

with open("/srv/scratch/PLM/results.csv", "a") as f:
    f.write(f"{args.esm_model}, {args.layer}, {args.dataset}, {test_loss}\n")
        
  

