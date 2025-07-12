import  matplotlib.pyplot  as  plt
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True) 
parser.add_argument('--layer', type=int, required=True)
parser.add_argument('--function', type=str, required=True)
args = parser.parse_args()

name = args.dataset[:-4]
data_predict = pd.read_csv(f"/srv/scratch/PLM/embeddings/{args.model}/{name}/layer_{args.layer}/predictions_{args.function}.csv")

true = list(data_predict['test'])
predict = list(data_predict['test_predicted'])
for i in range(len(true)):
    if abs(true[i]) < 1e-7:
        true = true[:i]
        predict = predict[:i]
        break

plt.figure(figsize=(8, 8))
plt.scatter(predict, true, c="royalblue")
plt.plot(predict, predict, "blue")
plt.title(f"plot of {args.model} for {name} on layer {args.layer}")
plt.xlabel("prediction")
plt.ylabel("true")


plt.savefig(f"plots/predictions_{name}.svg")


plt.figure(figsize=(10, 8))
data = pd.read_csv(f"/srv/scratch/PLM/datasets/{args.dataset}")
label = list(data['label'])
plt.hist(label, bins=100)
plt.title(f"histogram of {name}")


plt.savefig(f"plots/hist_{name}.svg")
