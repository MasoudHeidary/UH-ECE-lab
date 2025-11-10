import os
import math

CUDA = "cuda:0"     # cpu, cuda, cuda:#num
SEQ_LEN = 50        # sequence length
EPOCH = 30          # epoch to train for
VALIDATE = 0        # epoch to start validating

def clean_files():
    os.system("rm ./init_model.pth")
    os.system("rm ./__pycache__ -r")
    os.system("rm ./runs -r")

def restart_model():
    os.system("rm ./opus* -r")
    os.system("rm *.json")


def config(d_model, num_layer, seq_len, epoch, precision, cuda, validate_epoch, learning_rate):
    return f"{d_model} {num_layer} {seq_len} {epoch} '{precision}' {cuda} {validate_epoch} {learning_rate}"

def run_train(d_model, num_layer, seq_len, epoch, precision, cuda, validate, learning_rate):
    clean_files()
    restart_model()
    cfg = config(d_model, num_layer, seq_len, epoch, precision, cuda, validate, learning_rate)

    return not os.system(f'python train.py {cfg}')

def run_inference(d_model, num_layer, seq_len, precision, cuda):
    clean_files()
    cfg = config(d_model, num_layer, seq_len, 0, precision, cuda, 0, 0)

    return not os.system(f'python inference.py {cfg}')

def run_flops(d_model, num_layer, seq_len, precision, cuda):
    clean_files()
    cfg = config(d_model, num_layer, seq_len, 0, precision, cuda, 0, 0)

    return not os.system(f'python flops.py {cfg}')



"""
    explore optimal epoch [find best epoch for each configuration]
    train and validate for each single epoch
"""
# for d_model in [128, 256, 512, 1024]:
#     for num_layer in [1, 2, 4, 6, 8]:
#         run_train(d_model, num_layer, SEQ_LEN, EPOCH, "ftp32", CUDA, VALIDATE, 2e-5)
        # run_flops(d_model, num_layer, SEQ_LEN, "ftp32", CUDA)

        # run_inference(d_model, num_layer, 350, "ftp32", CUDA)
        # run_inference(d_model, num_layer, 350, "ftp16", CUDA)
        # run_inference(d_model, num_layer, 350, "ftp8", CUDA)

# run_train(128, 1, SEQ_LEN, 20, "ftp32", CUDA, VALIDATE)
# run_train(1024, 8, SEQ_LEN, 50, "ftp32", CUDA, VALIDATE)
# run_train(512, 6, SEQ_LEN, 50, "ftp32", CUDA, VALIDATE)


# best training epoch for each configuration
run_schedule = [
    # d_model, num_layer, epoch, learning_rate
    (128,   1,  30,     4e-5),
    (128,   2,  30,     4e-5),
    (128,   4,  30,     4e-5),
    (128,   6,  30,     4e-5),
    (128,   8,  30,     4e-5),

    (256,   1,  30,     4e-5),
    (256,   2,  30,     4e-5),
    (256,   4,  30,     4e-5),
    (256,   6,  30,     4e-5),
    (256,   8,  30,     4e-5),

    (512,   1,  19,     2e-5),
    (512,   2,  18,     2e-5),
    (512,   4,  21,     2e-5),
    (512,   6,  18,     2e-5),
    (512,   8,  18,     2e-5),

    (1024,   1,  10,    2e-5),
    (1024,   2,  11,    2e-5),
    (1024,   4,  8,     2e-5),
    (1024,   6,  8,     2e-5),
    (1024,   8,  10,    2e-5),
]

for run in run_schedule:
    d_model, num_layer, epoch, learning_rate = run
    epoch = 0
    run_train(d_model, num_layer, SEQ_LEN, epoch, "ftp32", CUDA, VALIDATE, learning_rate)
    
    for precision in ["ftp32", "ftp16", "ftp8"]:
        # run_inference(d_model, num_layer, SEQ_LEN, precision, CUDA)
        run_flops(d_model, num_layer, SEQ_LEN, precision, CUDA)
