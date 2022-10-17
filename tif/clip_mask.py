import glob
import os
import sys
import time
sys.path.append("I:/WHU_WY/Guangdong")
from osgeo import gdal
from osgeo import ogr
from shp.trans_shp import trans_shp
from shp.shp2raster import shp2raster
def searchShpByRaster(img_name, shp_dir):
    dataset = gdal.Open(img_name)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    transform = dataset.GetGeoTransform()
    left = transform[0]
    top = transform[3]
    right = left + transform[1] * im_width
    bottom = top + transform[5] * im_height

    used_shp = []
    shp_list = glob.glob(('{}*.shp'.format(shp_dir)))
    for shp_file in shp_list:
        input_shp = ogr.Open(shp_file)
        shp_layer = input_shp.GetLayer()
        x_min, x_max, y_min, y_max = shp_layer.GetExtent()
        flag = 0
        if y_min < top < y_max and x_min < left < x_max:
            flag = 1
        if y_min < bottom < y_max and x_min < left < x_max:
            flag = 1
        if y_min < top < y_max and x_min < right < x_max:
            flag = 1
        if y_min < bottom < y_max and x_min < right < x_max:
            flag = 1
        if bottom < y_min < top and left < x_min < right:
            flag = 1
        if bottom < y_max < top and left < x_min < right:
            flag = 1
        if bottom < y_min < top and left < x_max < right:
            flag = 1
        if bottom < y_max < top and left < x_max < right:
            flag = 1

        if flag == 1:
            used_shp.append(shp_file)
    return used_shp, transform[1]

def merge_shp(shp_list, file_name):
    files_string = " ".join(shp_list)
    print(files_string)
    shp_dir = os.path.join(os.path.dirname(shp_list[0]), file_name)
    if not os.path.exists(shp_dir):
        os.makedirs(shp_dir)
    command = "C:/Users/505/Anaconda3/envs/point_test/python.exe I:/WHU_WY/Guangdong/shp/ogrmerge.py -single -o {}/merged.shp ".format(shp_dir) + files_string
    print(os.popen(command).read())
    return shp_dir + "/merged.shp"

def clipMaskByImg(img_name, mask_name):
    dataset = gdal.Open(img_name)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    transform = dataset.GetGeoTransform()

    ms_dataset = gdal.Open(mask_name)
    ms_transform = ms_dataset.GetGeoTransform()
    if ms_transform[3] < transform[3]:
        transform_y = transform[3] + dataset.RasterYSize * transform[5]
    else:
        transform_y = transform[3]
    left_num = int((transform[0] - ms_transform[0]) / transform[1])
    bottom_num = int((transform_y - ms_transform[3]) / -transform[5])

    clip_mask_name = mask_name.replace(".tif", "_clip.tif")
    windows = [left_num, bottom_num, im_width, im_height]
    gdal.Translate(clip_mask_name, ms_dataset, srcWin=windows)
    a = 0

if __name__ == '__main__':
    shp_dir = "I:/WHU_WY/selsectSHP/"
    data_dir = "I:/WHU_WY/image/"
    file_list = glob.glob(('{}*.tif'.format(data_dir)))
    # for i, file in enumerate(file_list):
    # print("{}/{}".format(str(i + 1), str(len(file_list))))
    # 查询覆盖shp
    file = 'I:/WHU_WY/image_finish\\20DEC01030057-M2AS-013317249080_04_P002_FUS_DOM.tif'
    print(file)
    print("begin search")
    shp_list, pixel_size = searchShpByRaster(file, shp_dir)
    if len(shp_list) == 0:
        print("mistake appear")
        exit(0)
    print("finish search")
    # 合并覆盖shp
    print("begin merge shp")
    file_name = file.split("\\")[-1].split(".")[0]
    shp_file = merge_shp(shp_list, file_name)
    print("finish merge shp")
    # 类别转换
    print("begin translate shp")
    trans_shp(shp_file)
    print("finish translate shp")
    # shp转栅格
    print("begin shp2raster")
    output_raster = shp_file.split(".")[0] + '.tif'
    shp2raster(shp_file, output_raster, pixel_size)
    print("finish shp2raster")
    # 裁剪
    print("begin clip")
    clipMaskByImg(file, output_raster)
    print("finish clip")
    # clipMaskByImg("I:/WHU_WY/image/20APR28025614-M2AS-012339292070_05_P013_FUS_DOM.tif", "I:/WHU_WY/selsectSHP/20APR28025614-M2AS-012339292070_05_P013_FUS_DOM/merged.tif")
    # C:/Users/505/Anaconda3/envs/point_test/python.exe
    # 'I:/WHU_WY/image\\20DEC01030057-M2AS-013317249080_04_P002_FUS_DOM.tif'

    # 'I:/WHU_WY/image\\20JAN12025941-M2AS-012339292070_01_P006_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\20JAN15030939-M2AS-012339292070_01_P031_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\20JAN29031211-M2AS-012339292060_01_P005_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\20NOV01030658-M2AS-013246733010_02_P001_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\20NOV08030554-M2AS-013317249080_01_P010_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\20NOV09031120-M2AS-013317249080_01_P007_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\21JAN02031709-M2AS-013317249030_01_P002_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-01_20191108_L2A0000875654_1011900930030007_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-01_20201105_L2A0000982614_1012000800360065_C_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-01_20201112_L2A0000986358_1012000800610022_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-01_20201112_L2A0000986361_1012000800610020_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-01_20201112_L2A0000986372_1012000800610016_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-02_20201126_L2A0000991054_1012000800860003_C_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-03_20200429_L2A0000919406_1012000800070214_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-03_20201011_L2A0000982691_1012000800630064_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-03_20201221_L2A0001001564_1012000801090064_C_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-03_20201221_L2A0001001593_1012000801400050_C_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-04_20201126_L2A0000990973_1012000801400002_C_01-MUX_FUS_DOM.tif'
    # 'I:/WHU_WY/image\\SV1-04_20210101_L2A0001005493_1012000801240011_C_01-MUX_FUS_DOM.tif']


# SV1-01_20201112_L2A0000986361_1012000800610020_01-MUX_FUS_DOM.tif
# 20DEC01030059-M2AS-013317249080_04_P004_FUS_DOM.tif