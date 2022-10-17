import os
import torch
import numpy as np
import random
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.config import obtain_predict_args
from engine.predictor import Predictor
# 为每个卷积层搜索最适合它的卷积实现算法
# torch.backends.cudnn.benchmark=True

def main():
    args = obtain_predict_args()
    args.cuda = torch.cuda.is_available()

    print(args)
    torch.manual_seed(args.seed)
    predictor = Predictor(args)
    start = time.time()
    if args.mode == 'split':
        predictor.split_predict()
    if args.mode == 'concat':
        # predictor.concat_predict()
        predictor.ori_predict()
    end = time.time()
    print((end-start)/4)

if __name__ == "__main__":
    main()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
