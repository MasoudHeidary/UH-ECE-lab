# -*- coding: utf-8 -*-

from pathlib import Path

D_MODEL = 512
NUM_HEADS = 2
NUM_LAYERS = 2
SEQ_LEN = 350
LANG_SRC = 'en'
LANG_TGT = 'it'
NET_NAME = 'model_dense_enc'
NUM_EPOCHS = 1

def get_config():
    return {
        "net_name": f"{NET_NAME}",
        "batch_size": 8,
        "num_epochs": NUM_EPOCHS,
        "lr": 10**-4,
        "seq_len": SEQ_LEN,
        "d_models": [D_MODEL],
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "datasource": "opus_books",
        "lang_src": f"{LANG_SRC}",
        "lang_tgt": f"{LANG_TGT}",
        # "model_folder": "emb{d_model}_head{num_heads}_{lang_src}_{lang_tgt}_{net_name}",
        "model_folder": f"emb{D_MODEL}_head{NUM_HEADS}_{LANG_SRC}_{LANG_TGT}_{NET_NAME}",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }

def get_weights_file_path(config, epoch: str):
    # Format the model_folder string using the values from config
    model_folder = f"{config['datasource']}_{config['model_folder'].format(d_model=config['d_models'][0], num_heads=config['num_heads'], lang_src=config['lang_src'], lang_tgt=config['lang_tgt'], net_name=config['net_name'])}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

def latest_weights_file_path(config):
    # IMPORTANT: format the folder exactly like get_weights_file_path does
    fmt_folder = config["model_folder"].format(
        d_model=config["d_models"][0],
        num_heads=config["num_heads"],
        lang_src=config["lang_src"],
        lang_tgt=config["lang_tgt"],
        net_name=config["net_name"],
    )
    model_folder = f"{config['datasource']}_{fmt_folder}"
    weights_files = sorted(Path(model_folder).glob(f"{config['model_basename']}*"))
    if not weights_files:
        return None
    return str(weights_files[-1])
