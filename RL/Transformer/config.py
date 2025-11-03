# -*- coding: utf-8 -*-
"""
Language list:
'ca-de', 'ca-en', 'ca-hu', 'ca-nl', 'de-en', 'de-eo', 'de-es', 'de-fr', 'de-hu', 'de-it', 'de-nl', 
'de-pt', 'de-ru', 'el-en', 'el-es', 'el-fr', 'el-hu', 'en-eo', 'en-es', 'en-fi', 'en-fr', 'en-hu', 
'en-it', 'en-nl', 'en-no', 'en-pl', 'en-pt', 'en-ru', 'en-sv', 'eo-es', 'eo-fr', 'eo-hu', 'eo-it', 
'eo-pt', 'es-fi', 'es-fr', 'es-hu', 'es-it', 'es-nl', 'es-no', 'es-pt', 'es-ru', 'fi-fr', 'fi-hu', 
'fi-no', 'fi-pl', 'fr-hu', 'fr-it', 'fr-nl', 'fr-no', 'fr-pl', 'fr-pt', 'fr-ru', 'fr-sv', 'hu-it', 
'hu-nl', 'hu-no', 'hu-pl', 'hu-pt', 'hu-ru', 'it-nl', 'it-pt', 'it-ru', 'it-sv'
"""

from pathlib import Path
from log import Log
import sys



if (len(sys.argv) == 1):
    D_MODEL = 512
    NUM_HEADS = 2
    NUM_LAYERS = 6
    SEQ_LEN = 350
    LANG_SRC = 'en'
    LANG_TGT = 'it'
    NET_NAME = 'model_dense_enc'
    NUM_EPOCHS = 10
    PRECISION = "ftp32"
    DEVICE = "cuda:0"
    # log.println("running with default values")
else:
    D_MODEL     = int(sys.argv[1])
    NUM_HEADS   = 2
    NUM_LAYERS  = int(sys.argv[2])
    SEQ_LEN     = int(sys.argv[3])

    # High accuracy
    LANG_SRC =      'de'
    LANG_TGT =      'fr'

    # Fast Training
    # LANG_SRC    = 'el'
    # LANG_TGT    = 'en'
    NET_NAME    = 'model_dense_enc'
    NUM_EPOCHS  = 10
    PRECISION   = sys.argv[4]
    DEVICE      = sys.argv[5]

log = Log(f"output.{DEVICE}.log", terminal=True)


p = {
    'D_MODEL': D_MODEL, 'NUM_LAYERS': NUM_LAYERS, 
    'SEQ_LEN': SEQ_LEN, 'NUM_EPOCHS': NUM_EPOCHS,
    'PRECISION': PRECISION
}
log.println(f"{'@'*80}")
log.println(f"{'@'*80}")
log.println(f"{p}")


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
