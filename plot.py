import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--model', type=str, required=True) 
args = parser.parse_args()

df = pd.read_csv("/srv/scratch/PLM/results.csv")

def mse_plot(model):

    if model == "esm":
        models = ["esm_t6", "esm_t12", "esm_t30", "esm_t33"]
    elif model == "ankh":
        models = ["ankh-base", "ankh-large"]

    mse_values = {}
    spearman_values = {}

    for model in models:
        MSE = df[df['Embedding Model'] == model]['MSE']
        Spearman = df[df['Embedding Model'] == model]['Spearman']

        mse_values[model] = MSE
        spearman_values[model] = Spearman

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    color = plt.get_cmap("Set2")

    # Plot MSE (left y-axis)
    for i in range(len(models)):
        values = mse_values[models[i]]
        ax1.plot(values, label=f"{model} - MSE", color=color[i])
        values = spearman_values[models[i]]
        ax2.plot(values, label=f"{model} - Spearman", color=color[i])


    # Optional: Title and layout
    plt.title('MSE and Accuracy over layers')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f"/plots/{args.models}")

mse_plot(args.model)