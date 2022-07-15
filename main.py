import os
import torch
import numpy as np
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from config import obtain_retrain_args
from trainer import Trainer

# 为每个卷积层搜索最适合它的卷积实现算法
# torch.backends.cudnn.benchmark=True
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = obtain_retrain_args()
    args.cuda = torch.cuda.is_available()
    setup_seed(args.seed)
    trainer = Trainer(args)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)



if __name__ == "__main__":
    main()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
