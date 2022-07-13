from email.mime import base
import numpy as np
import cv2
import os
import random
from random import sample
random.seed(100)
GID_Vege_map = {"低矮植被": [0,255,0],
           "高大树木": [255, 0, 0],
           "其他": [153,102,51],
           "未知": [0,0,0]
        }
GID_15_map = {"工业用地": [200,0,0],
           "城市住宅": [250,0,150],
           "农村住宅": [200,150,150],
           "交通用地": [250,150,150],
           "水田": [0,200,0],
           "灌溉地": [150,250,0],
           "旱地": [150,200,150],
           "花园小区": [200,0,200],
           "乔木林地": [150,0,250],
           "灌木地": [150,150,250],
           "天然草地": [250,200,0],
           "人工草地": [200,200,0],
           "河流": [0,0,200],
           "湖泊": [0,150,200],
           "池塘": [0,200,250],
           "其他": [0,0,0],
        }
def translabel(map, label):
    h, w = label.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i, index in enumerate(map):
        rgb_image[label == i] = map[index]
    rgb_image[label == 255] = [0,0,0]

    return rgb_image

if __name__ == "__main__":
    # base_dir = "/media/dell/DATA/wy/data/512"
    # list_path = "/media/dell/DATA/wy/Seg_NAS/data/lists/GID/rs_test.lst"
    base_dir = "/media/dell/DATA/wy/data/GID-15/512"
    list_path = "/media/dell/DATA/wy/LightRS/data/list/gid15_vege_val.lst"
    with open(list_path, "rb") as f:
            datalist = f.readlines()

    label_list = [
                k[1]
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
    sample_list = label_list
    for label_path in sample_list:
        label = cv2.imread(os.path.join(base_dir, label_path), cv2.IMREAD_GRAYSCALE)
        # image = translabel(GID_map, label)
        image = translabel(GID_Vege_map, label)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filename = label_path.split('/')[-1]
        img_name= label_path.split('/')[-2]
        if not os.path.exists(os.path.join(base_dir, "rgb_label", img_name)):
            os.makedirs(os.path.join(base_dir, "rgb_label", img_name))
        cv2.imwrite(os.path.join(base_dir, "rgb_label", img_name, filename), image)
    
