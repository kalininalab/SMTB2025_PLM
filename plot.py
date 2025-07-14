import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Embeddings.")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--downstream', type=str, required=True) 

args = parser.parse_args()

def mse_plot(dataset, downstream):

    df = pd.read_csv("/srv/scratch/PLM/results_students.csv")
    models = ["esm_t6", "esm_t12", "esm_t30", "esm_t33", "ankh-base", "ankh-large"]

    mse_values = {}

    df = df[df["Dataset"] == args.dataset]
    df = df[df["Downstream Model"] == args.downstream]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    color = plt.get_cmap("Set2")

    for i in range(len(models)):
        m = models[i]
        # print(df[df['Embedding Model'] == m])
        if df[df['Embedding Model'] == m].empty:
            continue
        MSE = df[df['Embedding Model'] == m]['MSE']

        mse_values[m] = MSE

        layers = np.array(df[df['Embedding Model'] == m]["#layers"]).reshape(-1,1)
        scaler = MinMaxScaler()
        model = scaler.fit(layers)
        scaled_layers = model.transform(layers)

        values = mse_values[models[i]]
        ax1.plot(scaled_layers, values, label=f"{models[i]}", color=color(i), linestyle="--")


    # Optional: Title and layout
    plt.title(f'Matthews Correlation Coefficient over layers for {args.dataset}')
    ax1.set_xlabel('Normalized Layers')
    ax1.set_ylabel('Matthews Correlation Coefficient')

    ax1.legend(loc='lower center')

    fig.tight_layout()
    plt.savefig(f"deeploc/deeploc_bin.svg")

mse_plot(args.dataset, args.downstream)