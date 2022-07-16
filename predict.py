import os
import torch
import numpy as np
import random
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils.config import obtain_retrain_args
from engine.predictor import Predictor
# 为每个卷积层搜索最适合它的卷积实现算法
# torch.backends.cudnn.benchmark=True

def split_predict():
    args = obtain_retrain_args()
    args.cuda = torch.cuda.is_available()

    print(args)
    torch.manual_seed(args.seed)
    trainer = Predictor(args)
    # trainer.training(0)
    torch.cuda.synchronize()
    start = time.time()
    trainer.predict()
    torch.cuda.synchronize()
    end = time.time()
    print((end-start)/2100)

def concat_predict():
    args



if __name__ == "__main__":
    split_predict()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
