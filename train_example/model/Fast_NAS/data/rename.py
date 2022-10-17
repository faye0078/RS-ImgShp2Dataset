import os
import  glob

def rename(filepath):
    for filename0 in os.listdir(filepath):
        imgs = glob.glob('{}*.tif'.format(filepath + "/" + filename0 + "/"))
        for filename in imgs:
            if filename.split(".")[0].split("_")[-1] != "img":
                os.rename(filename, filename.split(".")[0] + "_img." + filename.split(".")[1])  # 重命名
rename("E:/wangyu_file/rs_Nas/src/data/datasets/VOCdevkit/512/test/image/")
# rename("E:/wangyu_file/rs_Nas/src/data/datasets/VOCdevkit/512/val/label/")