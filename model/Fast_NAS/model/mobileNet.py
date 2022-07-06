
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from nn.layer_factory import InvertedResidual, conv_bn_relu6, conv3x3


n_class = 5

class MobileNetV2(nn.Module):
    """MobileNetV2 definition"""

    # expansion rate, output channels, number of repeats, stride
    mobilenet_config = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    in_planes = 32  # number of input channels
    num_layers = len(mobilenet_config)

    def __init__(self, width_mult=1.0, return_layers=[1, 2, 4, 6]):
        super(MobileNetV2, self).__init__()
        self.return_layers = return_layers
        self.max_layer = max(self.return_layers)
        self.out_sizes = [
            self.mobilenet_config[layer_idx][1] for layer_idx in self.return_layers
        ]
        input_channel = int(self.in_planes * width_mult)
        self.layer1 = conv_bn_relu6(4, input_channel, 2)
        for layer_idx, (t, c, n, s) in enumerate(
                self.mobilenet_config[: self.max_layer + 1]
        ):
            output_channel = int(c * width_mult)
            features = []
            for i in range(n):
                if i == 0:
                    features.append(
                        InvertedResidual(input_channel, output_channel, s, t)
                    )
                else:
                    features.append(
                        InvertedResidual(input_channel, output_channel, 1, t)
                    )
                input_channel = output_channel

            setattr(self, "layer{}".format(layer_idx + 2), nn.Sequential(*features))
        # make it nn.Sequential
        # features.append(
        #
        # )
        self.outsize = nn.Sequential(
            nn.Conv2d(input_channel, 350, 1, 1, 0, bias=False),
            nn.BatchNorm2d(350),
            nn.ReLU6(inplace=True)
        )

        # building classifier

        self.classifier = conv3x3(input_channel, n_class, stride=1, bias=True)

        # self._initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        for layer_idx in range(7):
            x = getattr(self, "layer{}".format(layer_idx + 2))(x)
        # x = self.outsize(x)
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)
    model = nn.DataParallel(model).cuda()
    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url

        model.load_state_dict(torch.load("E:/wangyu_file/nas-segm-pytorch-master/src/ckpt/test.pth"), strict=False)
    return model

