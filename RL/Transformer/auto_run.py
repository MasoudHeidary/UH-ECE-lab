import os

def clean_files():
    os.system("rm ./init_model.pth")
    # os.system("rm ./__pycache__ -r")
    os.system("rm ./runs -r")

def restart_model():
    os.system("rm ./opus* -r")
    os.system("rm *.json")

def run_train(d_model, num_layers, seq_len, precision, cuda_device):
    clean_files()
    restart_model()
    os.system(f'python train.py {d_model} {num_layers} {seq_len} "{precision}" {cuda_device}')

def run_inference(d_model, num_layers, seq_len, precision, cuda_device):
    clean_files()
    os.system(f'python inference.py {d_model} {num_layers} {seq_len} "{precision}" {cuda_device}')

def run_flops(d_model, num_layers, seq_len, precision, cuda_device):
    clean_files()
    os.system(f'python flops.py {d_model} {num_layers} {seq_len} "{precision}" {cuda_device}')


run_train(512, 6, 350, "ftp32", "cuda:1")
run_inference(512, 6, 350, "ftp32", "cuda:1")
run_flops(512, 6, 350, "ftp32", "cuda:1")
# run_inference(512, 6, 350, "ftp16", "cuda:0")
# run_inference(512, 6, 350, "ftp8", "cuda:0")
# run_inference(512, 6, 350, "int8", "cuda:0")
# run_inference(512, 6, 350, "ftp4", "cuda:0")
# run_inference(512, 6, 350, "int4", "cuda:0")


run_train(512, 6, 250, "ftp32", "cuda:1")
run_inference(512, 6, 250, "ftp32", "cuda:1")
run_flops(512, 6, 250, "ftp32", "cuda:1")