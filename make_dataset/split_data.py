from PIL import Image
import glob
import numpy as np
import os
from osgeo import gdal
import tifffile as tiff

def is_exist_zero_pixel_4(array):
    a = np.where(np.all(array == [0, 0, 0, 0], axis=-1))[:2]
    if len(a[0]) >50:
        return True
    else:
        return False   
    
def is_exist_zero_pixel_1(array):
    num = sum(sum(array == 0))
    if num > 2049:
        return True
    else:
        return False        
            

def image_clip(img_path, size):

    img_name = img_path.split('.')[-2].split('/')[-1]
    img_dir = "/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/sar/" + img_name
    folder = os.path.exists(img_dir)
    if not folder:
        os.makedirs(img_dir)

    imarray = gdal.Open(img_path).ReadAsArray()
    imarray = np.transpose(imarray, (1, 2, 0))
    imshape = imarray.shape
    H = imshape[0]
    W = imshape[1]
    num_col = int(W / size[1]) - 1
    num_row = int(H / size[0]) - 1
    step_col = (W - num_col * size[1]) - size[1]
    step_row = (H - num_row * size[0]) - size[0]

    for row in range(num_row):
        for col in range(num_col):
            clipArray = imarray[row * size[0]:(row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
            if is_exist_zero_pixel_4(clipArray):
                continue
            img_filepath = img_dir + '/' + img_name + "_" + str(
                row + 1) + "_" + str(col + 1) + "_img.tif"

            tiff.imwrite(img_filepath, clipArray)
            print("row: ", row, "col: ", col)

    for row in range(num_row):
        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
        if is_exist_zero_pixel_4(clipArray):
                continue
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 1) + "_img.tif"
        tiff.imwrite(img_filepath, clipArray)

        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
        if is_exist_zero_pixel_4(clipArray):
                continue
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('1drong!!')

        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 2) + "_img.tif"
        tiff.imwrite(img_filepath, clipArray)

    for col in range(num_col):
        clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
        if is_exist_zero_pixel_4(clipArray):
                continue
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(col + 1) + "_img.tif"
        tiff.imwrite(img_filepath, clipArray)

        clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, col * size[1]:(col + 1) * size[1]]
        if is_exist_zero_pixel_4(clipArray):
                continue
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('2drong!!')

        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(col + 1) + "_img.tif"
        tiff.imwrite(img_filepath, clipArray)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
    if not is_exist_zero_pixel_4(clipArray):
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(num_col + 1) + "_img.tif"
        tiff.imwrite(img_filepath, clipArray)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if not is_exist_zero_pixel_4(clipArray):
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('3drong!!')
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(num_col + 2) + "_img.tif"
        tiff.imwrite(img_filepath, clipArray)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1]:(num_col + 1) * size[1]]
    if not is_exist_zero_pixel_4(clipArray):
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('4drong!!')
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(num_col + 1) + "_img.tif"
        tiff.imwrite(img_filepath, clipArray)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if not is_exist_zero_pixel_4(clipArray):
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('5drong!!')
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('6drong!!')
        
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(num_col + 2) + "_img.tif"
        tiff.imwrite(img_filepath, clipArray)
def label_clip(img_path, size, type):

    img_name = img_path.split('.')[-2].split('/')[-1]
    img_dir = "/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/" + type + "/" + img_name
    folder = os.path.exists(img_dir)
    if not folder:
        os.makedirs(img_dir)
    if "sar" in img_path:
        imarray = gdal.Open(img_path).ReadAsArray().astype(np.uint16)
    else:
        imarray = gdal.Open(img_path).ReadAsArray()
    imshape = imarray.shape
    H = imshape[0]
    W = imshape[1]
    num_col = int(W / size[1]) - 1
    num_row = int(H / size[0]) - 1
    step_col = (W - num_col * size[1]) - size[1]
    step_row = (H - num_row * size[0]) - size[0]

    for row in range(num_row):
        for col in range(num_col):
            clipArray = imarray[row * size[0]:(row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
            if is_exist_zero_pixel_1(clipArray) and type!="sar":
                continue
            clipImg = Image.fromarray(clipArray)

            img_filepath = img_dir + '/' + img_name + "_" + str(
                row + 1) + "_" + str(col + 1) + "_{}.png".format(type)
            clipImg.save(img_filepath)

    for row in range(num_row):
        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
        if is_exist_zero_pixel_1(clipArray) and type!="sar":
                continue
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 1) + "_{}.png".format(type)
        clipImg.save(img_filepath)

        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
        if is_exist_zero_pixel_1(clipArray) and type!="sar":
                continue
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('1drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 2) + "_{}.png".format(type)
        clipImg.save(img_filepath)

    for col in range(num_col):
        clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
        if is_exist_zero_pixel_1(clipArray) and type!="sar":
                continue
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(col + 1) + "_{}.png".format(type)
        clipImg.save(img_filepath)

        clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, col * size[1]:(col + 1) * size[1]]
        if is_exist_zero_pixel_1(clipArray) and type!="sar":
                continue
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('2drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(col + 1) + "_{}.png".format(type)
        clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
    if is_exist_zero_pixel_1(clipArray) and type!="sar":
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(num_col + 1) + "_{}.png".format(type)
        clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if is_exist_zero_pixel_1(clipArray) and type!="sar":
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('3drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(num_col + 2) + "_{}.png".format(type)
        clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1]:(num_col + 1) * size[1]]
    if is_exist_zero_pixel_1(clipArray) and type!="sar":
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('4drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(num_col + 1) + "_{}.png".format(type)
        clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if is_exist_zero_pixel_1(clipArray) and type!="sar":
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('5drong!!')
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('6drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(num_col + 2) + "_{}.png".format(type)
        clipImg.save(img_filepath)
        
def fliter_data(img_path, label_path, label3_path, sar_path):
    for dir, _, file_list in os.walk(img_path):
        for file in file_list:
            dir_name = dir.split('/')[-1]
            image_full_path = os.path.join(dir, file)
            label_full_path = os.path.join(label_path, dir_name, file).replace('img', 'label').replace('tif', 'png')
            label3_full_path = os.path.join(label3_path, dir_name, file).replace('img', 'label3').replace('tif', 'png')
            sar_full_path = []
            sar_base = os.path.join(sar_path, dir_name, file).replace('img', 'sar').replace('tif', 'png')
            for i in range(12):
                each_sar_path = sar_base.replace(dir_name, str(i + 1) + "_" + dir_name)
                sar_full_path.append(each_sar_path)
            
            flag = True
            if not os.path.exists(label_full_path) or not os.path.exists(label3_full_path):
                flag = False
            
            for sar in sar_full_path:
                if not os.path.exists(sar):
                    flag = False
                        
            if not flag:
                if os.path.exists(image_full_path):
                    os.remove(image_full_path)
                if os.path.exists(label_full_path):
                    os.remove(label_full_path)
                if os.path.exists(label3_full_path):
                    os.remove(label3_full_path)
                for sar in sar_full_path:
                    if os.path.exists(sar):
                        os.remove(sar)            
            
            
def remove_otherfiles(img_path, label_path, label3_path, sar_path):
    # get all img
    all_img_list = []
    for dir, _, file_list in os.walk(img_path):
        for file in file_list:
            all_img_list.append(file)
    print("all img num: ", len(all_img_list))
    for dir, _, file_list in os.walk(label_path):
        for file in file_list:
            img_file = file.replace("label", "img").replace("png", "tif")
            if not img_file in all_img_list:
                os.remove(os.path.join(dir, file))
                a = 0
            print("finish{}".format(img_file))
                
    for dir, _, file_list in os.walk(label3_path):
        for file in file_list:
            img_file = file.replace("label3", "img").replace("png", "tif")
            if not img_file in all_img_list:
                os.remove(os.path.join(dir, file))
                a = 0
                
    for dir, _, file_list in os.walk(sar_path):
        for file in file_list:
            dir_name = dir.split('/')[-1]
            replace_str = dir_name.split('_')[1]
            img_file = file.replace(dir_name, replace_str).replace("sar", "img").replace("png", "tif")
            if not img_file in all_img_list:
                os.remove(os.path.join(dir, file))
                a = 0
            
if __name__=='__main__':
    # split iamge
    # folder = os.path.exists("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/image/")
    # if not folder:
    #     os.makedirs("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/image/")

    # img_dir = '/media/dell/DATA/wy/data/guiyang/剑河/光学影像2021/'
    # imgs = glob.glob('{}*.png'.format(img_dir))
    # for img in imgs:
    #     print(img)
    #     image_clip(img, [1024, 1024])
        
    # split label
    # folder = os.path.exists("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/label/")
    # if not folder:
    #     os.makedirs("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/label/")

    # img_dir = '/media/dell/DATA/wy/data/guiyang/剑河/数据集/'
    # folder = "/media/dell/DATA/wy/data/guiyang/剑河/数据集/"
    # for dir, _, files in os.walk(folder):
    #     for file in files:
    #         img = dir.split('/')[-2]
    #         type = dir.split('/')[-1]
    #         if type == 'builtup_add':
    #             print(os.path.join(dir, file))
    #             label_clip(os.path.join(dir, file), [1024, 1024]m, "label")
                
    # split label3
    # folder = os.path.exists("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/label3/")
    # if not folder:
    #     os.makedirs("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/label3/")

    # img_dir = '/media/dell/DATA/wy/data/guiyang/剑河/数据集/'
    # folder = "/media/dell/DATA/wy/data/guiyang/剑河/数据集/"
    # for dir, _, files in os.walk(folder):
    #     for file in files:
    #         img = dir.split('/')[-2]
    #         type = dir.split('/')[-1]
    #         if type == 'label3_clip':
    #             print(os.path.join(dir, file))
    #             label_clip(os.path.join(dir, file), [1024, 1024], "label3")
                
    # split sar
    # folder = os.path.exists("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/sar/")
    # if not folder:
    #     os.makedirs("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/sar/")

    # img_dir = '/media/dell/DATA/wy/data/guiyang/剑河/数据集/'
    # folder = "/media/dell/DATA/wy/data/guiyang/剑河/数据集/"
    # for dir, _, files in os.walk(folder):
    #     for file in files:
    #         img = dir.split('/')[-2]
    #         type = dir.split('/')[-1]
    #         if type == 'sar_clip' and "resample" not in file:
    #             print(os.path.join(dir, file))
    #             label_clip(os.path.join(dir, file), [1024, 1024], "sar")
    
    # fliter data
    # fliter_data("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/image/", "/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/label/", "/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/label3/", "/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/sar/")
    remove_otherfiles("/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/image/", "/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/label/", "/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/label3/", "/media/dell/DATA/wy/data/guiyang/剑河/数据集/裁剪/sar/")
    # folder = os.path.exists("../../../data/GID-15/512/label")
    # if not folder:
    #     os.makedirs("../../data/GID-15/512/image")
    #
    # img_dir = '../../data/Large-scale Classification_5classes/image_NirRGB/'
    # imgs = glob.glob('{}*.tif'.format(img_dir))
    # for img in imgs:
    #     image_clip(img, [512, 512])

