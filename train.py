import os
import torch
import numpy as np
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from utils.config import obtain_retrain_args
from engine.trainer import Trainer

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
