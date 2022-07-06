import numpy as np
import torch.nn as nn
import torch.distributed as dist

from model.operations import NaiveBN, ABN
from model.aspp import ASPP
from model.decoder import Decoder, Decoder1
from model.new_model import get_default_arch, newModel


class Retrain_Autodeeplab(nn.Module):
    def __init__(self, args):
        super(Retrain_Autodeeplab, self).__init__()
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        BatchNorm2d = ABN if args.use_ABN else NaiveBN
        if (not args.dist and args.use_ABN) or (args.dist and args.use_ABN and dist.get_rank() == 0):
            print("=> use ABN!")
        if args.network_arch is not None and args.cell_arch is not None:
            network_path, cell_arch, network_arch = np.load(args.network_path,allow_pickle=True), np.load(args.cell_arch,allow_pickle=True), np.load(args.network_arch,allow_pickle=True)
        else:
            network_arch, cell_arch, network_path = get_default_arch()
        # network_arch =
        self.encoder = newModel(network_arch, cell_arch, args.num_classes, len(network_path), args.filter_multiplier, BatchNorm=BatchNorm2d, args=args)
        self.aspp0 = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[0][-1]],
                         256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        self.aspp1 = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[1][-1]],
                          256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        self.aspp2 = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[2][-1]],
                          256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        self.aspp3 = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[network_path[3][-1]],
                          256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        self.decoder = Decoder1(args.num_classes, filter_multiplier=args.filter_multiplier * args.block_multiplier,
                               args=args, last_level=network_path[-1])

    def forward(self, x):
        low_features, encoder_output = self.encoder(x)
        features = []
        for i in range(len(encoder_output)):
            features.append(getattr(self, "aspp{}".format(i))(encoder_output[i]))

        decoder_output = self.decoder(low_features, features)
        # return nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(decoder_output)
        return decoder_output
    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params