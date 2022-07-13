import os
import numpy as np
from PIL import Image
import cv2

dir = "/media/dell/DATA/wy/data/GID-15/512/"
data_file = '/media/dell/DATA/wy/LightRS/data/list/gid15_vege5_val.lst'
with open(data_file, "rb") as f:
    datalist = f.readlines()
try:
    # self.datalist = [
    #     (k, k.replace('image/', 'label/').replace('img.tif', 'label.png'))
    #     for k in map(
    #         lambda x: x.decode("utf-8").strip("\n").strip("\r"), datalist
    #     )
    # ]
    datalist = [
        (k[0], k[1])
        for k in map(
            lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
        )
    ]
except ValueError:  # Adhoc for test.
    datalist = [
        (k, k) for k in map(lambda x: x.decode("utf-8").strip("\n"), datalist)
    ]

for data in datalist:
    img_name = os.path.join(dir, data[0])
    image = np.asarray(Image.open(img_name), dtype=np.float64)
    ndvi = (image[:, :, 0] - image[:, :, 3]) / (image[:, :, 0] + image[:, :, 3])
    result = np.zeros(image.shape[:2])
    result[ndvi > 0] = 255
    img = image[:,:,1:]
    cv2.imwrite('./label.png', result)
    cv2.imwrite('./image.png', img)
    # ndvi = (image[:, :, 0] - image[:, :, 2]) / (image[:, :, 0] + image[:, :, 2])