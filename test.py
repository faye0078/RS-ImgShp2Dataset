import os
import torch
from ptflops import get_model_complexity_info
from thop import profile
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.config import obtain_retrain_args
from engine.trainer import Trainer
from torchsummary import summary
from torchstat import stat


def main():
    args = obtain_retrain_args()
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    device = torch.device("cpu")
    model = trainer.model.to(device)

    stat(model, (5, 512, 512))

    macs, params = get_model_complexity_info(trainer.model, (4, 512, 512), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    print("this model macs: " + macs)
    print("this model params: " + params)

    trainer.validation(0)

if __name__ == "__main__":
    main()
