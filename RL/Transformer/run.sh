#!/bin/bash
# train, evaluate and get FLOPS of the model

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm "./__pycache__" -r
rm ./runs -r
rm ./opus* -r
rm ./init_model.pth

python ./train.py