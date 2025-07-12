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
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

name = args.dataset[:-4]

#Download the model
from transformers import AutoTokenizer, T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained(f"ElnaggarLab/{args.model}")
model = T5EncoderModel.from_pretrained(f"ElnaggarLab/{args.model}").to(device)

# Download the file
df = pd.read_csv(f"/srv/scratch/PLM/datasets/{args.dataset}")

def embeddings(x):
    
    seq = df.at[x, 'Sequence']
    ID = df.at[x, 'ID']

    tokens = tokenizer.encode(seq, return_tensors="pt")
        
    embeddings = model(tokens.to(device), output_hidden_states=True)

    for i in range(0,49):
        a = embeddings.hidden_states[i]
        b = a[0, 1:]
        c = torch.mean(b, 0)
        directory = Path(f"/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{i}/")
        directory.mkdir(exist_ok=True, parents=True)
        with open(f"{directory}/{ID}.pkl", "wb") as f:
            pickle.dump(c, f)

for y in tqdm(range(len(df))):
    embeddings(y)