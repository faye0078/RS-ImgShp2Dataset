"""Main file for search.

KD from RefineNet-Light-Weight-152 (args.do_kd => keep in memory):
  Task0 - pre-computed
  Task1 - on-the-fly

Polyak Averaging (args.do_polyak):
  Task0 - only decoder
  Task1 - encoder + decoder

Search:
  Task0 - task0_epochs - validate every epoch
  Task1 - task1_epochs - validate every epoch

"""
import pyximport
pyximport.install()
# general libs
import argparse
import logging
import os
import random
import time
import numpy as np

# pytorch libs
import torch
import torch.nn as nn

# custom libs
from data.loaders import create_loaders, create_test_loader
from engine.inference import validate, test_validate
from engine.trainer import populate_task0, train_task0, train_segmenter
from helpers.utils import (
    apply_polyak,
    compute_params,
    init_polyak,
    load_ckpt,
    Saver,
    TaskPerformer,
)
from nn.encoders import create_encoder

from utils.train_default_args import *
from utils.solvers import create_optimisers
from helpers.utils import AverageMeter

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


class Segmenter(nn.Module):
    """Create Segmenter"""

    def __init__(self, encoder, decoder):
        super(Segmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    # Set-up experiment
    args = get_arguments()

    logger = logging.getLogger(__name__)
    logger.debug(args)

    args.num_tasks = len(args.num_classes)
    segm_crit = nn.NLLLoss2d(ignore_index=255).cuda()

    # Set-up random seeds
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Initialise encoder
    encoder = create_encoder(ctrl_version=args.ctrl_version,)



    logger.info(
        " Loaded Encoder with #TOTAL PARAMS={:3.2f}M".format(
            compute_params(encoder)[0] / 1e6
        )
    )

    # Generate teacher if any （定义）
    kd_net = True
    kd_crit = None
    args.do_kd = False
    if args.do_kd:
        from kd.rf_lw.model_lw_v2 import rf_lw152 as kd_model

        kd_crit = nn.MSELoss().cuda()
        kd_net = (
            kd_model(pretrained=True, num_classes=args.num_classes[0]).cuda().eval()
        )
        logger.info(
            " Loaded teacher, #TOTAL PARAMS={:3.2f}M".format(
                compute_params(kd_net)[0] / 1e6
            )
        )


    def create_segmenter(encoder):
        if args.ctrl_version == "cvpr":
            from nn.micro_decoders import MicroDecoder as Decoder
        elif args.ctrl_version == "wacv":
            from nn.micro_decoders import TemplateDecoder as Decoder
        with torch.no_grad():

            # decoder_config = [[6, [0, 0, 9, 6], [0, 0, 10, 9], [0, 5, 4, 6]], [[2, 3], [0, 0], [0, 2]]] # first_train_config
            # decoder_config = [[2, [0, 0, 10, 4], [3, 1, 9, 8], [1, 2, 2, 1]], [[3, 1], [4, 4], [2, 0]]] # second_train_config
            # decoder_config = [[9, [0, 0, 2, 4], [3, 2, 1, 6], [3, 6, 10, 0]], [[1, 3], [2, 2], [3, 4]]] # third_train_config
            # decoder_config = [[9, [0, 0, 2, 7], [2, 2, 10, 1], [3, 1, 4, 4]], [[3, 0], [1, 4], [4, 1]]] # forth_train_config

            # decoder_config = [[0, [0, 0, 10, 10], [0, 3, 5, 4], [5, 2, 5, 6]], [[1, 2], [1, 3], [0, 1]]] # 0.934
            # decoder_config = [[3, [0, 0, 3, 2], [1, 0, 1, 3], [4, 3, 3, 4]], [[0, 3], [4, 2], [1, 2]]] # 0.936
            # decoder_config = [[4, [0, 0, 4, 10], [3, 1, 10, 1], [6, 0, 2, 7]], [[2, 3], [2, 0], [3, 4]]] # 0.935
            decoder_config = [[9, [0, 0, 8, 5], [3, 1, 7, 10], [0, 3, 2, 7]], [[0, 0], [0, 1], [4, 2]]] # 0.934

            decoder = Decoder(
                inp_sizes=encoder.out_sizes,
                num_classes=args.num_classes[0],
                config=decoder_config,
                agg_size=args.agg_cell_size,
                aux_cell=args.aux_cell,
                repeats=args.sep_repeats,
            )

        # Fuse encoder and decoder
        segmenter = nn.DataParallel(Segmenter(encoder, decoder)).cuda()
        logger.info(
            " Created Segmenter, #PARAMS (Total, No AUX)={}".format(
                compute_params(segmenter)
            )
        )
        return segmenter, decoder_config

    # Sample first configuration
    segmenter, decoder_config = create_segmenter(encoder)
    # hrnet_decoder_path = "E:/wangyu_file/model/ach1_hrnet.pth"
    #
    # segmenter.load_state_dict(
    #     torch.load(hrnet_decoder_path), strict=False
    # )
    logger.info(" Decoder: {}".format(decoder_config))

    del encoder

    # Create dataloaders
    train_loader, val_loader, do_search = create_loaders(args)
    test_loader = create_test_loader(args)

    # Initialise task performance measurers
    task_ps = [
        [
            TaskPerformer(maxval=0.01, delta=0.9)
            for _ in range(args.num_segm_epochs[idx] // args.val_every[idx])
        ]
        for idx, _ in enumerate(range(args.num_tasks))
    ]


    logger.info(" Pre-computing data for task0")
    Xy_train = populate_task0(segmenter, train_loader, kd_net, args.n_task0, args.do_kd)
    if args.do_kd:
        del kd_net

    logger.info(" Training Process Starts")
    avr = [AverageMeter(), AverageMeter()]


    start = time.time()
    torch.cuda.empty_cache()
    logger.info(" Training Segmenter")
    best_miou = 0.9

    for task_idx in range(args.num_tasks):
        # task_idx = 1
        if task_idx == 1:
            del Xy_train
        torch.cuda.empty_cache()
        # Change dataloader
        train_loader.batch_sampler.batch_size = args.batch_size[task_idx]
        for loader in [train_loader, val_loader]:
            try:
                loader.dataset.set_config(
                    crop_size=args.crop_size[task_idx],
                    shorter_side=args.shorter_side[task_idx],
                )
            except AttributeError:
                # for subset

                loader.dataset.set_config(
                    crop_size=args.crop_size[task_idx],
                    resize_side=args.resize_side[task_idx],
                )

        logger.info(" Training Task {}".format(str(task_idx)))
        # Optimisers
        optim_enc, optim_dec = create_optimisers(
            args.enc_optim,
            args.dec_optim,
            args.enc_lr[task_idx],
            args.dec_lr[task_idx],
            args.enc_mom[task_idx],
            args.dec_mom[task_idx],
            args.enc_wd[task_idx],
            args.dec_wd[task_idx],
            segmenter.module.encoder.parameters(),
            segmenter.module.decoder.parameters(),
        )
        avg_param = init_polyak(
            args.do_polyak, segmenter.module.decoder if task_idx == 0 else segmenter
        )

        for epoch_segm in range(args.num_segm_epochs[task_idx]):
            if task_idx == 0:
                train_task0(
                    Xy_train,
                    segmenter,
                    optim_dec,
                    epoch_segm,
                    segm_crit,
                    kd_crit,
                    args.batch_size[0],
                    args.freeze_bn[0],
                    args.do_kd,
                    args.kd_coeff,
                    args.dec_grad_clip,
                    args.do_polyak,
                    avg_param=avg_param,
                    polyak_decay=0.9,
                    aux_weight=args.dec_aux_weight,
                )
            else:

                torch.cuda.empty_cache()
                train_segmenter(
                    segmenter,
                    train_loader,
                    optim_enc,
                    optim_dec,
                    epoch_segm,
                    segm_crit,
                    args.freeze_bn[1],
                    args.enc_grad_clip,
                    args.dec_grad_clip,
                    args.do_polyak,
                    args.print_every,
                    aux_weight=args.dec_aux_weight,
                    avg_param=avg_param,
                    polyak_decay=0.99,
                )
                _, params = compute_params(segmenter)
                logger.info(
                    " mobel_total, #TOTAL PARAMS={:3.2f}M".format(
                        params / 1e6
                    )
                )
            apply_polyak(
                args.do_polyak,
                segmenter.module.decoder if task_idx == 0 else segmenter,
                avg_param,
            )
            if (epoch_segm + 1) % (args.val_every[task_idx]) == 0:
                logger.info(
                    " Validating Segmenter, Task {}".format(
                         str(task_idx)
                    )
                )
                task_miou = validate(
                    segmenter,
                    val_loader,
                    0,
                    epoch_segm,
                    num_classes=args.num_classes[task_idx],
                    print_every=args.print_every,
                    omit_classes=args.val_omit_classes,
                )
                # Verifying if we are continuing training this architecture.
                c_task_ps = task_ps[task_idx][
                    (epoch_segm + 1) // args.val_every[task_idx] - 1
                    ]
                print(task_miou)
                print(avr[task_idx].avg)
                if task_idx == 1 and task_miou > best_miou:
                    best_miou = task_miou
                    PATH = "E:/wangyu_file/model/ach4_hrnet.pth"
                    torch.save(segmenter.state_dict(),PATH)
                    logger.info(
                        " current best val miou={}".format(
                            task_miou
                        )
                    )
                    test_miou = test_validate(
                        segmenter,
                        test_loader,
                        num_classes=5,
                        print_every=10,
                    )



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()
