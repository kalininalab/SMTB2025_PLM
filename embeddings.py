import pandas as pd
import numpy as np
import torch
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--esm_model', type=str, required=True)

args = parser.parse_args()

n = args.esm_model[6:8]
if "_" in n:
    n = args.esm_model[6:7]
n = int(n)

name = args.dataset[:-4]

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.esm_model}")
model = AutoModelForMaskedLM.from_pretrained(f"facebook/{args.esm_model}")

# Download the file
df = pd.read_csv(f"/srv/scratch/PLM/datasets/{args.dataset}")

data = df["Sequence"]
ID = df["ID"]

def embeddings(x):
    
    seq = df.at[x, 'Sequence']
    ID = df.at[x, 'ID']
    tokens = tokenizer.encode(seq, return_tensors="pt")
    embeddings = model(tokens, output_hidden_states=True)

    means = []

    for i in range(0,n+1):
        a = embeddings.hidden_states[i]
        b = a[0, 1:-1, :]
        c = torch.mean(b, 0)
        means.append(c)
        directory = Path(f"/srv/scratch/PLM/embeddings/{n}/{name}/layer_{i}/")
        directory.mkdir(exist_ok=True, parents=True)
        with open(f"{directory}/{ID}.pkl", "wb") as f:
            pickle.dump(c, f)



for y in tqdm(range(len(data))):
    embeddings(y)