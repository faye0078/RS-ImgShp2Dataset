"""Data Transformations and pre-processing."""

from __future__ import print_function, division

import os
import numpy as np
import torch
from osgeo import gdal
from PIL import Image
from torch.utils.data import Dataset

class Guangdong(Dataset):
    """Custom Pascal VOC"""

    def __init__(self, stage, data_file, data_dir, transform_trn=None, transform_val=None, transform_test=None):
        """
        Args:
            data_file (string): Path to the data file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        if stage == 'predict':
            self.datalist = [d.decode("utf-8").strip().split(" ") for d in datalist]
        else:
            self.datalist = [
                (k[0], k[1])
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.stage = stage
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def set_config(self, crop_size, resize_side):
        self.transform_trn.transforms[0].resize_side = resize_side
        self.transform_trn.transforms[2].crop_size = crop_size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        print("begin read data")
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        dataset=gdal.Open(img_name)    #打开文件
        im_width = dataset.RasterXSize  #栅格矩阵的列数
        im_height = dataset.RasterYSize  #栅格矩阵的行数

        transform = dataset.GetGeoTransform() #仿射矩阵
        projection = dataset.GetProjection() #地图投影信息
        image = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
        image_0 = image.transpose(1, 2, 0)
        del image
        image_1 = image_0 / 255.0 # TODO：这里是否需要除什么？
        del image_0
        image_2 = image_1 - self.mean
        del image_1
        image = image_2 / self.std
        del image_2
        print("read data finish")
        if self.stage == 'predict':
            image_1 = image.transpose((2, 0, 1))
            del image
            image = torch.from_numpy(image_1)
            del image_1
            sample = {"image": image,
                        "transform": transform,
                        "projection": projection,
                        "size": (im_width, im_height),
                        "name": self.datalist[idx][0]}
            return sample
        else:
            mask = np.zeros((im_height, im_width)) # TODO: 读tif的mask
            sample = {"image": image, 
                        "mask": mask, 
                        "transform": transform,
                        "projection": projection,
                        "size": (im_width, im_height),
                        "name": self.datalist[idx][1]}

        return sample
