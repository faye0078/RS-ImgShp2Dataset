
import nn.encoders
from data.train_loaders import create_loaders, create_test_loader
import argparse
from helpers.utils import AverageMeter, try_except
from helpers.utils import (
    apply_polyak,
    compute_params,
    init_polyak,
    load_ckpt,
    Saver,
    TaskPerformer,
)
from engine.inference import validate, test_validate

from model.seg_hrnet import get_seg_model
from model.mobileNet import mobilenet_v2
from model.build_autodeeplab import Retrain_Autodeeplab
from model.re_train_autodeeplab import obtain_retrain_autodeeplab_args
from utils.train_default_args import *
from nn.encoders import create_encoder
from nn.micro_decoders import MicroDecoder as Decoder
from main_search import Segmenter
import logging
import torch
import torch.nn as nn
import time
import numpy as np
import random
import os
from tqdm import tqdm
n_class = 5
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NAS Search")

    # Dataset
    parser.add_argument(
        "--train-dir",
        type=str,
        default=TRAIN_DIR,
        help="Path to the training set directory.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=VAL_DIR,
        help="Path to the validation set directory.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=TEST_DIR,
        help="Path to the test set directory."
    )
    parser.add_argument(
        "--train-list",
        type=str,
        default=TRAIN_LIST,
        help="Path to the training set list.",
    )
    parser.add_argument(
        "--val-list",
        type=str,
        default=VAL_LIST,
        help="Path to the validation set list.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default=TEST_LIST,
        help="Path to the test set list."
    )
    parser.add_argument(
        "--meta-train-prct",
        type=int,
        default=META_TRAIN_PRCT,
        help="Percentage of examples for meta-training set.",
    )
    parser.add_argument(
        "--resize-side",
        type=int,
        nargs="+",
        default=RESIZE_SIDE,
        help="Resize side transformation.",
    )
    parser.add_argument(
        "--resize-longer-side",
        action="store_true",
        help="Whether to resize the longer side when doing the resize transformation.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs="+",
        default=CROP_SIZE,
        help="Crop size for training,",
    )
    parser.add_argument(
        "--normalise-params",
        type=list,
        default=NORMALISE_PARAMS,
        help="Normalisation parameters [scale, mean, std],",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=BATCH_SIZE,
        help="Batch size to train the segmenter model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for pytorch's dataloader.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        nargs="+",
        default=NUM_CLASSES,
        help="Number of output classes for each task.",
    )
    parser.add_argument(
        "--low-scale",
        type=float,
        default=LOW_SCALE,
        help="Lower bound for random scale",
    )
    parser.add_argument(
        "--high-scale",
        type=float,
        default=HIGH_SCALE,
        help="Upper bound for random scale",
    )
    parser.add_argument(
        "--n-task0",
        type=int,
        default=N_TASK0,
        help="Number of images per task0 (trainval)",
    )
    parser.add_argument(
        "--val-resize-side",
        type=int,
        default=VAL_RESIZE_SIDE,
        help="Resize side transformation during validation.",
    )
    parser.add_argument(
        "--val-crop-size",
        type=int,
        default=VAL_CROP_SIZE,
        help="Crop size for validation.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=VAL_BATCH_SIZE,
        help="Batch size to validate the segmenter model.",
    )
    parser.add_argument(
        "--val-omit-classes",
        type=int,
        nargs="*",
        default=VAL_OMIT_CLASSES,
        help="Classes to omit in the validation.",
    )

    # Encoder
    parser.add_argument(
        "--enc-grad-clip",
        type=float,
        default=ENC_GRAD_CLIP,
        help="Clip norm of encoder gradients to this value.",
    )

    # Decoder
    parser.add_argument(
        "--dec-grad-clip",
        type=float,
        default=DEC_GRAD_CLIP,
        help="Clip norm of decoder gradients to this value.",
    )
    parser.add_argument(
        "--dec-aux-weight",
        type=float,
        default=DEC_AUX_WEIGHT,
        help="Auxiliary loss weight for each aggregate head.",
    )

    # General
    parser.add_argument(
        "--freeze-bn",
        type=bool,
        nargs="+",
        default=FREEZE_BN,
        help="Whether to keep batch norm statistics intact.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of epochs to train for the controller.",
    )
    parser.add_argument(
        "--num-segm-epochs",
        type=int,
        nargs="+",
        default=NUM_SEGM_EPOCHS,
        help="Number of epochs to train for each sampled network.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=PRINT_EVERY,
        help="Print information every often.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed to provide (near-)reproducibility.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=SNAPSHOT_DIR,
        help="Path to directory for storing checkpoints.",
    )
    parser.add_argument(
        "--ckpt-path", type=str, default=CKPT_PATH, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--val-every",
        nargs="+",
        type=int,
        default=VAL_EVERY,
        help="How often to validate current architecture.",
    )
    parser.add_argument(
        "--summary-dir", type=str, default=SUMMARY_DIR, help="Summary directory."
    )

    # Optimisers
    parser.add_argument(
        "--enc-lr",
        type=float,
        nargs="+",
        default=ENC_LR,
        help="Learning rate for encoder.",
    )
    parser.add_argument(
        "--dec-lr",
        type=float,
        nargs="+",
        default=DEC_LR,
        help="Learning rate for decoder.",
    )
    parser.add_argument(
        "--ctrl-lr", type=float, default=CTRL_LR, help="Learning rate for controller."
    )
    parser.add_argument(
        "--enc-mom",
        type=float,
        nargs="+",
        default=ENC_MOM,
        help="Momentum for encoder.",
    )
    parser.add_argument(
        "--dec-mom",
        type=float,
        nargs="+",
        default=DEC_MOM,
        help="Momentum for decoder.",
    )
    parser.add_argument(
        "--enc-wd",
        type=float,
        nargs="+",
        default=ENC_WD,
        help="Weight decay for encoder.",
    )
    parser.add_argument(
        "--dec-wd",
        type=float,
        nargs="+",
        default=DEC_WD,
        help="Weight decay for decoder.",
    )
    parser.add_argument(
        "--enc-optim",
        type=str,
        default=ENC_OPTIM,
        help="Optimiser algorithm for encoder.",
    )
    parser.add_argument(
        "--dec-optim",
        type=str,
        default=DEC_OPTIM,
        help="Optimiser algorithm for decoder.",
    )
    parser.add_argument(
        "--do-kd",
        type=bool,
        default=DO_KD,
        help="Whether to do knowledge distillation (KD).",
    )
    parser.add_argument(
        "--kd-coeff", type=float, default=KD_COEFF, help="KD loss coefficient."
    )
    parser.add_argument(
        "--do-polyak",
        type=bool,
        default=DO_POLYAK,
        help="Whether to do Polyak averaging.",
    )

    # Controller
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=LSTM_HIDDEN_SIZE,
        help="Number of neurons in the controller's RNN.",
    )
    parser.add_argument(
        "--lstm-num-layers",
        type=int,
        default=LSTM_NUM_LAYERS,
        help="Number of layers in the controller.",
    )
    parser.add_argument(
        "--num-ops", type=int, default=NUM_OPS, help="Number of unique operations."
    )
    parser.add_argument(
        "--num-agg-ops",
        type=int,
        default=NUM_AGG_OPS,
        help="Number of unique operations.",
    )
    parser.add_argument(
        "--agg-cell-size",
        type=int,
        default=AGG_CELL_SIZE,
        help="Common size inside decoder",
    )
    parser.add_argument(
        "--ctrl-baseline-decay",
        type=float,
        default=CTRL_BASELINE_DECAY,
        help="Baseline decay.",
    )
    parser.add_argument(
        "--ctrl-version",
        type=str,
        choices=["cvpr", "wacv"],
        default=CTRL_VERSION,
        help="Type of microcontroller",
    )
    parser.add_argument(
        "--ctrl-agent",
        type=str,
        choices=["ppo", "reinforce"],
        default=CTRL_AGENT,
        help="Gradient estimator algorithm",
    )
    parser.add_argument(
        "--dec-num-cells",
        type=int,
        default=DEC_NUM_CELLS,
        help="Number of cells to apply.",
    )
    parser.add_argument(
        "--cell-num-layers",
        type=int,
        default=CELL_NUM_LAYERS,
        help="Number of branches inside the learned cell.",
    )
    parser.add_argument(
        "--cell-max-repeat",
        type=int,
        default=CELL_MAX_REPEAT,
        help="Maximum number of repeats of the sampled cell",
    )
    parser.add_argument(
        "--cell-max-stride",
        type=int,
        default=CELL_MAX_STRIDE,
        help="Maximum stride of the sampled cell",
    )
    parser.add_argument(
        "--aux-cell",
        type=bool,
        default=AUX_CELL,
        help="Whether to use the cell design in-place of auxiliary cell.",
    )
    parser.add_argument(
        "--sep-repeats",
        type=int,
        default=SEP_REPEATS,
        help="Number of repeats inside Sep Convolution.",
    )
    return parser.parse_args()

@try_except
def train_model(
    net,
    train_loader,
    optim,
    epoch,
    segm_crit,
    freeze_bn,
    aux_weight=-1,
    print_every=10,
):

    try:
        train_loader.dataset.set_stage("train")
    except AttributeError:
        train_loader.dataset.dataset.set_stage("train")  # for subset
    net.train()
    # freeze_bn = True
    # if freeze_bn:
    #     for m in mobilenet.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    tbar = tqdm(train_loader, desc='\r')

    for i, sample in enumerate(tbar):
        start = time.time()
        image = sample["image"].float().cuda()
        target = sample["mask"].cuda()
        target_var = torch.autograd.Variable(target).float()
        # Compute output
        output = net(image)

        if isinstance(output, tuple):
            output, aux_outs = output

        target_var = nn.functional.interpolate(
            target_var[:, None], size=output.size()[2:], mode="nearest"
        ).long()[:, 0]

        soft_output = nn.LogSoftmax()(output)
        # Compute loss and backpropagate
        loss = segm_crit(soft_output, target_var)

        if aux_weight > 0:
            for aux_out in aux_outs:
                aux_out = nn.Upsample(
                    size=target_var.size()[1:], mode="bilinear", align_corners=False
                )(aux_out)
                aux_out = nn.LogSoftmax()(aux_out)
                # Compute loss and backpropagate
                loss += segm_crit(aux_out, target_var) * aux_weight

        optim.zero_grad()

        loss.backward()

        optim.step()

        losses.update(loss.item())
        batch_time.update(time.time() - start)

        logger = logging.getLogger(__name__)

        tbar.set_description('loss: %.3f' % (losses.avg))


def main():

    args = get_arguments()
    logger = logging.getLogger(__name__)
    logger.debug(args)

    # Set-up random seeds
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # net = mobilenet_v2(False)
    # net = get_seg_model()
    auto_args = obtain_retrain_autodeeplab_args() 
    auto_args.num_classes = 5
    net = Retrain_Autodeeplab(auto_args)
    net = net.cuda()

    # encoder = create_encoder(ctrl_version=args.ctrl_version,)

    # hrnet 4 layers
    # decoder_config = [[8, [0, 0, 9, 1], [2, 1, 10, 8], [4, 1, 3, 9]], [[0, 3], [3, 2], [4, 1]]] #pretrain_model_config
    # decoder_config = [[6, [0, 0, 9, 6], [0, 0, 10, 9], [0, 5, 4, 6]], [[2, 3], [0, 0], [0, 2]]] # first_train_config
    # decoder_config = [[9, [0, 0, 2, 7], [2, 2, 10, 1], [3, 1, 4, 4]], [[3, 0], [1, 4], [4, 1]]] # forth_train_config\

    #mobilenet
    # decoder_config = [[0, [0, 0, 10, 10], [0, 3, 5, 4], [5, 2, 5, 6]], [[1, 2], [1, 3], [0, 1]]] # 0.934
    # decoder_config = [[3, [0, 0, 3, 2], [1, 0, 1, 3], [4, 3, 3, 4]], [[0, 3], [4, 2], [1, 2]]] # 0.936
    # decoder_config = [[4, [0, 0, 4, 10], [3, 1, 10, 1], [6, 0, 2, 7]], [[2, 3], [2, 0], [3, 4]]] # 0.935
    # decoder_config = [[9, [0, 0, 8, 5], [3, 1, 7, 10], [0, 3, 2, 7]], [[0, 0], [0, 1], [4, 2]]] # 0.934

    # hrnet 7 layers

    # decoder_config = [[6, [0, 0, 6, 8], [3, 3, 3, 4], [3, 4, 10, 4]], [[1, 4], [2, 0], [0, 2]]] # 0.8956
    # decoder = Decoder(
    #     inp_sizes=[256,48,48,48,96,192,384],
    #     num_classes=args.num_classes[0],
    #     config=decoder_config,
    #     agg_size=args.agg_cell_size,
    #     aux_cell=args.aux_cell,
    #     repeats=args.sep_repeats,
    # )
    # net = nn.DataParallel(Segmenter(encoder, decoder)).cuda()

    logger.info(
        " Loaded Encoder with #TOTAL PARAMS={:3.2f}M".format(
            compute_params(net)[0] / 1e6
        )
    )

    train_loader, val_loader, do_search = create_loaders(args)
    test_loader = create_test_loader(args)
    segm_crit = nn.NLLLoss2d(ignore_index=255).cuda()


    best_miou = 0


    for epoch in range(100):
        optim = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        train_loader.batch_sampler.batch_size = args.batch_size[0]

        train_model(
            net,
            train_loader,
            optim,
            epoch,
            segm_crit,
            freeze_bn = False,
            aux_weight= -1,
            print_every= 200,
        )
        if (epoch + 1) % (1) == 0:
            logger.info(
                " Validating hrnet decoder epoch{}".format(
                    str(epoch)
                )
            )
            task_miou = validate(
                net,
                val_loader,
                1,
                epoch,
                num_classes=5,
                print_every=100,
                omit_classes=[0],
            )
            if task_miou > best_miou:
                # PATH = "E:/wangyu_file/nas-segm-pytorch-master/src/ckpt/check_point.pth"
                PATH = "../model/best_multipath_ach.pth"
                torch.save(net.state_dict(),PATH)
                logger.info(
                    " current best val miou={}".format(
                        task_miou
                    )
                )
                best_miou = task_miou
                test_miou = test_validate(
                    net,
                    test_loader,
                    num_classes=5,
                    print_every=100,
                )


def load_test():

    # net = get_seg_model()

    args = get_arguments()
    logger = logging.getLogger(__name__)
    logger.debug(args)

    encoder = create_encoder(ctrl_version=args.ctrl_version,)
    # decoder_config = [[9, [0, 0, 8, 5], [3, 1, 7, 10], [0, 3, 2, 7]], [[0, 0], [0, 1], [4, 2]]]
    decoder_config = [[6, [0, 0, 6, 8], [3, 3, 3, 4], [3, 4, 10, 4]], [[1, 4], [2, 0], [0, 2]]] 
    decoder = Decoder(
        inp_sizes= [256,48,48,48,96,192,384],
        num_classes=args.num_classes[0],
        config=decoder_config,
        agg_size=args.agg_cell_size,
        aux_cell=args.aux_cell,
        repeats=args.sep_repeats,
    )
    net = nn.DataParallel(Segmenter(encoder, decoder)).cuda()
    # net = get_seg_model()
    model_paths = "../model/best_hrnet_ach.pth"
    net.load_state_dict(
        torch.load(model_paths), strict=False
    )


    test_loader = create_test_loader(args)

    train_loader, val_loader, do_search = create_loaders(args)
    task_miou = validate(
        net,
        val_loader,
        1,
        0,
        num_classes=5,
        print_every=10,
        omit_classes=[0],
    )
    test_miou = test_validate(
        net,
        test_loader,
        num_classes=5,
        print_every=100,
    )
    logger.info(
        " current best val miou={}".format(
            test_miou
        )
    )
    # PATH = "E:/wangyu_file/model/pretrained_mobile_encoder.pth"
    # torch.save(net.module.encoder.state_dict(),PATH)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    # load_test()
    main()
