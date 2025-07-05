# SMTB 2025 -- Lab16 -- ESM Project

The goal of this project is to extract the embeddings of PLMs like ESM2-t6 at all layers. We then want to train small neural networks on these embeddings to predict properties of proteins. This will give us insights into in which layers the model has learned/stored which properties of the proteins.

## Project Structure

The project will be structured modularly. This means, we will have three types of scripts to be called and executed consecutively to go from datasets to performances. Therefore, it is important that the scripts produce the output in the same fashion as they are expected by the next script.

In the following, we will describe the scripts based on the fluorescence dataset and the ESM-t6 model, but the same structure applies to all datasets and models.

### 1. **Data Preparation**

This script will download the dataset file from huggingface, extract the relevant columns (amino acid sequence and label(s)), and save it in a CSV format.

```shell
python fluorescence.py --save-path /srv/data/plm//datasets
```

### 2. **Embedding Extraction**

Scripts that extract embeddings from the pre-trained models (like ESM2-t6) for the prepared datasets. These scripts will handle the loading of the model, processing the data, and saving the embeddings in the dedicated directory.

```shell
python esm.py --data-path /srv/data/plm/datasets/fluorescence.csv --output-path /srv/data/plm/embeddings/esm_t6/fluorescence/ --num-layers 6
```

### 3. **Training**

A script that trains a small neural network on the extracted embeddings from one layer on one dataset to predict a specific property of proteins. These script will load the embeddings, define the neural network architecture, and train the model.

```shell
python mlp.py --input-dim 320 --hidden-dim 64 --output-dim 1 --mode regression --data-path /srv/data/plm/datasets/fluorescence.csv --embeds-path /srv/data/plm/embeddings/esm_t6/fluorescence/layer_0/ --log-folder ../test_logs/
```

## Data-directory structure

This is the intended structure for us to store the results of each module.

```shell
/srv/data/plm/
   ├── datasets/
   │   ├── fluorescence.csv
   │   └── ... (other datasets)
   └── embeddings/
       ├── esm_t6/
       │   ├── fluorescence/
       │   │   ├── layer_0/
       |   │   │   ├── P00000.pkl
       |   │   │   └── ... (more embeddings)
       │   │   └── ... (more layers)
       │   └── ... (other datasets)
       └── ... (other models)
```
