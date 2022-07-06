import torch
import torch.nn as nn
import torch.nn.functional as F
from model.operations import NaiveBN

ALIGN_CORNERS = None
class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm=NaiveBN):
        super(Decoder, self).__init__()
        self.last_layer = nn.Sequential(nn.Conv2d(1024,256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x):

        x0_h, x0_w = x[0].size(2) * 4, x[0].size(3) * 4
        x0 = F.interpolate(x[0], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x0, x1, x2, x3], 1)
        x = self.last_layer(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder1(nn.Module):
    def __init__(self, num_classes, filter_multiplier, BatchNorm=NaiveBN, args=None, last_level=0):
        super(Decoder1, self).__init__()

        self.encoder_last_layer = nn.Sequential(nn.Conv2d(1024,256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.Dropout(0.1),
                                       )

        self.lowfeatures_last_layer = nn.Sequential(nn.Conv2d(1024,256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.Dropout(0.1),
                                       )

        low_level_inplanes = filter_multiplier
        C_low = 48
        self.conv1 = nn.Conv2d(1024, C_low, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.last_conv = nn.Sequential(nn.Conv2d(688,256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):

        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x0 = F.interpolate(x[0], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x0, x1, x2, x3], 1)
        # x = self.last_layer(x)
        x0_h, x0_w = low_level_feat[0].size(2) * 4, low_level_feat[0].size(3) * 4
        low_level_feat0 = F.interpolate(low_level_feat[0], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        low_level_feat1 = F.interpolate(low_level_feat[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        low_level_feat2 = F.interpolate(low_level_feat[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        low_level_feat3 = F.interpolate(low_level_feat[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        low_level_feat = torch.cat([low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3], 1)

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()