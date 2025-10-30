

from train import *


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    cfg = get_config()
    get_flops(cfg, log=log)
