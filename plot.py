import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True) 
parser.add_argument('--downstream', type=str, required=True) 

args = parser.parse_args()

def mse_plot(model, dataset, downstream):

    df = pd.read_csv("/srv/scratch/PLM/results.csv")
    if model == "esm":
        models = ["esm_t6", "esm_t12", "esm_t30", "esm_t33"]
    elif model == "ankh":
        models = ["ankh-base", "ankh-large"]

    mse_values = {}
    spearman_values = {}

    df = df[df["Dataset"] == args.dataset]
    df = df[df["Downstream Model"] == args.downstream]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    color = plt.get_cmap("Set2")

    for i in range(len(models)):
        m = models[i]
        print(df[df['Embedding Model'] == m])
        MSE = df[df['Embedding Model'] == m]['MSE']
        Spearman = df[df['Embedding Model'] == m]['Spearman']

        mse_values[m] = MSE
        spearman_values[m] = Spearman

        layers = np.array(df[df['Embedding Model'] == m]["#layers"]).reshape(-1,1)
        scaler = MinMaxScaler()
        model = scaler.fit(layers)
        scaled_layers = model.transform(layers)

        values = mse_values[models[i]]
        ax1.plot(scaled_layers, values, label=f"{models[i]}", color=color(i), linestyle="--")
        values = spearman_values[models[i]]
        ax2.plot(scaled_layers, values, label=f"{models[i]}", color=color(i))


    # Optional: Title and layout
    plt.title(f'MSE and Spearman over layers for {args.dataset}')
    ax1.set_xlabel('Normalized Layers')
    ax1.set_ylabel('MSE')
    ax2.set_ylabel('Spearman')
    model_handles, model_labels = ax1.get_legend_handles_labels()

    # Manual legend for line styles
    line_spearman = Line2D([0], [0], color='black', linestyle='-', label="Spearman")
    line_mse = Line2D([0], [0], color='black', linestyle='--', label="MSE")

    # Final legend: 4 models + 2 line-style indicators
    final_handles = model_handles + [line_mse, line_spearman]
    final_labels = model_labels + ["MSE", "Spearman"]

    # Set legend
    ax1.legend(handles=final_handles, labels=final_labels, loc='upper right')

    fig.tight_layout()
    plt.savefig(f"plots/{args.model}")

mse_plot(args.model, args.dataset, args.downstream)