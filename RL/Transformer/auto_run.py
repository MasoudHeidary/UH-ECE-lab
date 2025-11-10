import os
import math

CUDA = "cuda:0"
SEQ_LEN = 50
EPOCH = 30
VALIDATE = 1

def clean_files():
    os.system("rm ./init_model.pth")
    os.system("rm ./__pycache__ -r")
    os.system("rm ./runs -r")

def restart_model():
    os.system("rm ./opus* -r")
    os.system("rm *.json")

def run_train(d_model, num_layers, seq_len, epoch, precision, cuda_device, validate):
    clean_files()
    restart_model()
    os.system(f'python train.py {d_model} {num_layers} {seq_len} {epoch} "{precision}" {cuda_device} {validate}')

def run_inference(d_model, num_layers, seq_len, precision, cuda_device):
    clean_files()
    os.system(f'python inference.py {d_model} {num_layers} {seq_len} "{precision}" {cuda_device}')

def run_flops(d_model, num_layers, seq_len, precision, cuda_device):
    clean_files()
    os.system(f'python flops.py {d_model} {num_layers} {seq_len} "{precision}" {cuda_device}')



"""
    explore optimal epoch [find best epoch for each configuration]
    train and validate for each single epoch
"""
# for d_model in [512, 1024]:
#     for num_layer in [1, 2, 4, 6, 8]:
#         run_train(d_model, num_layer, SEQ_LEN, EPOCH, "ftp32", CUDA, VALIDATE)
        # run_flops(d_model, num_layer, 350, "ftp32", CUDA)

        # run_inference(d_model, num_layer, 350, "ftp32", CUDA)
        # run_inference(d_model, num_layer, 350, "ftp16", CUDA)
        # run_inference(d_model, num_layer, 350, "ftp8", CUDA)

# run_train(128, 1, SEQ_LEN, 20, "ftp32", CUDA, VALIDATE)
# run_train(1024, 8, SEQ_LEN, 50, "ftp32", CUDA, VALIDATE)
# run_train(512, 6, SEQ_LEN, 50, "ftp32", CUDA, VALIDATE)


"""
    explore optimal epoch [find best epoch for each configuration]
    train each configuration for best number of epochs -> different precision inference -> accuracy & FLOPs
"""
run_schedule = [
    # (d_model, num_layer, training_epoch)
    (128, 1, 9),
    (128, 2, 9),
    (128, 4, 9),
    (128, 6, 9),
    (128, 8, 9),

    (256, 1, 6),
    (256, 2, 8),
    (256, 4, 9),
    (256, 6, 9),
    (256, 8, 9),

    (512, 1, 4),
    (512, 2, 5),
    (512, 4, 6),
    # (512, 6, 9),
    # (512, 8, 9),

    (1024, 1, 3),
    (1024, 2, 5),
    (1024, 4, 10),
    (1024, 6, 10),
    (1024, 8, 5),
]

def perplex(loss):
    return math.exp(loss)

model_output = [
    # (d_model, num_layer, perplexity)
    (128,   1,  50, perplex(4.1348569798091103)),
    (128,   2,  50, perplex(4.1004396940550590)),
    (128,   4,  50, perplex(4.0568744091932564)),
    (128,   6,  50, perplex(3.9957100833274103)),
    (128,   8,  49, perplex(3.9606015085007857)),

    (256,   1,  42, perplex(3.8412323128781449)),
    (256,   2,  42, perplex(3.7895274472408650)),
    (256,   4,  41, perplex(3.7633226452761398)),
    (256,   6,  38, perplex(3.7453280548512677)),
    (256,   8,  32, perplex(3.7196653452408090)),

    (512,   1,  19, perplex(3.4155236363927006)),
    (512,   2,  18, perplex(3.3781511771558512)),
    (512,   4,  21, perplex(3.3559833365945417)),
    (512,   6,  19, perplex(3.3407655107682575)),
    (512,   8,  17, perplex(3.3327261963551180)),

    (1024,  1,  11, perplex(3.4094154320409977)),
    (1024,  2,  11, perplex(3.3658961472229180)),
    (1024,  4,  8,  perplex(3.3402649931405360)),
    (1024,  6,  8,  perplex(3.3268537445673867)),
    (1024,  8,  10, perplex(3.2918238428110340)),
]





