#! bin/bash/

CUDA_VISIBLE_DEVICES=0,1,2 python3 ../main.py -m supervised.training.train_num=128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048,2176,2304
5