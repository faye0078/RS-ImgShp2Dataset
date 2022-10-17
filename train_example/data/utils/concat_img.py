import numpy as np
import cv2
from collections import OrderedDict
import os
def concat(dirname, imgs, imgsize):
    imshape = [512,512,3]
    H = imshape[0]
    W = imshape[1]
    num_col = int(imgsize[1] / W)
    num_row = int(imgsize[0] / H)
    step_col = num_col * W
    step_row = num_row * H

    sum_img = np.zeros(imgsize)
    for row in range(num_row):
        for col in range(num_col):
            img_name = dirname + "_{}_{}_label.png".format(row + 1, col + 1)
            sum_img[row * H:(row + 1) * H, col * W:(col + 1) * W] = imgs[img_name]
    for row in range(num_row):
        img_name = dirname + "_{}_{}_label.png".format(row + 1, 15)
        sum_img[row * H:(row + 1) * H, step_col:imgsize[1]] = imgs[img_name][:, 512 - imgsize[1] + step_col: 512]
    for col in range(num_col):
        img_name = dirname + "_{}_{}_label.png".format(14, col + 1)
        sum_img[step_row:imgsize[0], col * W:(col + 1) * W] = imgs[img_name][512 - imgsize[0] + step_row: 512, :]

    img_name = dirname + "_{}_{}_label.png".format(14, 15)
    sum_img[step_row:imgsize[0], step_col:imgsize[1]] = imgs[img_name][512 - imgsize[0] + step_row: 512, 512 - imgsize[1] + step_col: 512]

    return sum_img


label_dir = "/media/dell/DATA/wy/LightRS/run/GID-Vege5/predict/PIDNet/experiment_0/"
for dirpath, dirnames, filenames in os.walk(label_dir):
    imgs = OrderedDict()
    if len(filenames) == 0:
        continue 
    for filename in filenames:
        label_path = os.path.join(dirpath, filename)
        label = cv2.imread(label_path)
        imgs[filename] = label
    sum_img = concat(dirpath.split("/")[-1], imgs, [6800, 7200,3])
    cv2.imwrite('/media/dell/DATA/wy/LightRS/run/GID-Vege5/predict/PIDNet/' + dirpath.split("/")[-1] + '.png' , sum_img)

