"""Data Transformations and pre-processing."""

from __future__ import print_function, division

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class GIDVege5(Dataset):
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
        try:
            # self.datalist = [
            #     (k, k.replace('image/', 'label/').replace('img.tif', 'label.png'))
            #     for k in map(
            #         lambda x: x.decode("utf-8").strip("\n").strip("\r"), datalist
            #     )
            # ]
            self.datalist = [
                (k[0], k[1])
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
        except ValueError:  # Adhoc for test.
            self.datalist = [
                (k, k) for k in map(lambda x: x.decode("utf-8").strip("\n"), datalist)
            ]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.stage = stage
        self.mean = (0.485, 0.456, 0.406, 0.411)
        self.std = (0.229, 0.224, 0.225, 0.227)


    def set_config(self, crop_size, resize_side):
        self.transform_trn.transforms[0].resize_side = resize_side
        self.transform_trn.transforms[2].crop_size = crop_size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])

        image = np.asarray(Image.open(img_name), dtype=np.float64)
        ndvi = (image[:, :, 0] - image[:, :, 3]) / ((image[:, :, 0] + image[:, :, 3])+0.0001)
        ndvi[ndvi>1] = 1
        ndvi[ndvi<-1] = -1
        ndvi = np.expand_dims(ndvi, axis = 2)
        image = image / 255.0
        image = image-self.mean
        image = image / self.std

        image = np.concatenate((image, ndvi), axis=2)
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
