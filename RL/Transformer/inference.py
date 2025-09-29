# -*- coding: utf-8 -*-
# File: transformer_bo/inference.py
from train import *


if __name__ == '__main__':
    log.println(f"inference:")
    warnings.filterwarnings("ignore")

    cfg = get_config()
    inference_model(cfg, log=log, precision=PRECISION)
    # get_flops(cfg, log=log)
    