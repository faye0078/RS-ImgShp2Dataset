import glob
import os
import collections
from my_functions import *
def caculate_all_label_distribution(img_dir):
    all_label_list = glob.glob(os.path.join(img_dir, "*.tif"))
    all_dis = collections.Counter()
    for i, label_path in enumerate(all_label_list):
        all_dis += caculate_distribution(label_path)
        print("finish {}/{}".format(str(i+1), str(len(all_label_list))))
    return all_dis

if __name__ == '__main__':
    jianhe = caculate_all_label_distribution("/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/数据集/v1/label/")
    xixiu = caculate_all_label_distribution("/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/数据集/v1/xixiu/label/")
    all = jianhe+xixiu
    print(all)
    sum = 0
    for index in all:
        sum +=  all[index]
    for index in all:
        all[index] = all[index]/sum
        
    print('\n')
    print(all)