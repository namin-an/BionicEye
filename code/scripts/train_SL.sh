#! bin/bash/

CUDA_VISIBLE_DEVICES=0,1,2 python3 ../main.py -m supervised.training.seed=32,64,128,256,512,1024,2048
