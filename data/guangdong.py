"""Data Transformations and pre-processing."""

from __future__ import print_function, division

import os
import numpy as np
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
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.25, 0.25, 0.25)

    def set_config(self, crop_size, resize_side):
        self.transform_trn.transforms[0].resize_side = resize_side
        self.transform_trn.transforms[2].crop_size = crop_size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])

        image = np.asarray(Image.open(img_name), dtype=np.float64)
        image = image / 255.0 # TODO：这里是否需要除什么？
        image = image - self.mean
        image = image / self.std

        sample = {"image": image, "name": self.datalist[idx][1]}
        if self.stage == 'predict':
            if self.transform_test:
                sample = self.transform_test(sample)
            return sample

        mask = np.array(Image.open(msk_name))
        # if img_name != msk_name:
        #     assert len(mask.shape) == 2, "Masks must be encoded without colourmap"
        sample = {"image": image, "mask": mask, "name": self.datalist[idx][1]}
        if self.stage == "train":
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == "val":
            if self.transform_val:
                sample = self.transform_val(sample)
        elif self.stage == 'test':
            if self.transform_test:
                sample = self.transform_test(sample)
        return sample
