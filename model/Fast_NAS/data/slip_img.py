try:
    import Image
    import ImageDraw
except:
    from PIL import Image
    from PIL import ImageDraw
import glob
import numpy as np
import os
import sys


def image_clip(img_path, size):

    # 转换为数组进行分割操作，计算能完整分割的行数、列数
    imarray = np.array(Image.open(img_path))
    imshape = imarray.shape
    image_col = int(imshape[1]/size[1])
    image_row = int(imshape[0]/size[0])

    img_name= img_path.split(".")[0].split("\\")[1]

    # 两个for循环分割能完整分割的图像，并保存图像、坐标转换文件
    for row in range(image_row):
        for col in range(image_col):
            clipArray = imarray[row*size[0]:(row+1)*size[0],col*size[1]:(col+1)*size[1]]
            clipImg = Image.fromarray(clipArray)
            folder = os.path.exists("E:/wangyu_file/GID/Fine Land-cover Classification_15classes/image_RGB/clip")
            # 判断文件夹是否存在，不存在则新建国家文件
            if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs("E:/wangyu_file/GID/Fine Land-cover Classification_15classes/image_RGB/clip")  # makedirs 创建文件时如果路径不存在会创建这个路径
            img_filepath = 'E:/wangyu_file/GID/Fine Land-cover Classification_15classes/image_RGB/clip/' + img_name + "_" +str(row) + "_" + str(col) + ".tif"
            clipImg.save(img_filepath)

if __name__=='__main__':
    img_dir = 'E:/wangyu_file/GID/Fine Land-cover Classification_15classes/image_RGB/'
    # img_dir = 'E:/wangyu_file/GID/Fine Land-cover Classification_15classes/label_15classes/'
    imgs = glob.glob('{}*.tif'.format(img_dir))
    for img in imgs:
        image_clip(img, [512, 512])

