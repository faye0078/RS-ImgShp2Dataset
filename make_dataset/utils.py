import os
from collections import OrderedDict
from osgeo import gdal
def get_id_names(id, file_list):
    id_name_list = []
    for file in file_list:
        if file.split("_")[1] == id:
            id_name_list.append(file)
    return id_name_list

def get_all_type_file(dir, file_type):
    all_file_list = []
    for dir, _, file_list in os.walk(dir):
        for file in file_list:
            if file.endswith(file_type):
                all_file_list.append(os.path.join(dir, file))
    
    return all_file_list

def get_label1_color_table():
    tb = gdal.ColorTable()
    tb.SetColorEntry(1, (117, 219, 87, 255))
    tb.SetColorEntry(2, (0, 128, 0, 255))
    tb.SetColorEntry(3, (255,255,0, 255))
    tb.SetColorEntry(4, (219, 95, 87, 255))
    tb.SetColorEntry(5, (0, 98, 255, 255))
    tb.SetColorEntry(6, (153, 93, 19, 255))
    return tb
def get_label2_color_table():
    tb = gdal.ColorTable()
    tb.SetColorEntry(1, (117, 219, 87, 255))
    tb.SetColorEntry(2, (0, 128, 0, 255))
    tb.SetColorEntry(3, (255,255,0, 255))
    tb.SetColorEntry(4, (219, 95, 87, 255))
    tb.SetColorEntry(5, (0, 98, 255, 255))
    tb.SetColorEntry(6, (153, 93, 19, 255))
    tb.SetColorEntry(7, (219, 208, 87, 255))
    return tb

def get_label3_color_table():
    tb = gdal.ColorTable()
    tb.SetColorEntry(1, (88, 176, 167, 255))
    tb.SetColorEntry(2, (123, 196, 123, 255))
    tb.SetColorEntry(3, (173, 219, 87, 255))
    tb.SetColorEntry(4, (117, 219, 87, 255))
    tb.SetColorEntry(5, (0, 128, 0, 255))
    tb.SetColorEntry(6, (255,255,0, 255))
    tb.SetColorEntry(7, (219, 95, 87, 255))
    tb.SetColorEntry(8, (0, 98, 255, 255))
    tb.SetColorEntry(9, (153, 93, 19, 255))
    tb.SetColorEntry(10, (219, 208, 87, 255))
    return tb

a = ['/media/dell/DATA/wy/data/guiyang/数据集/v2/image/89_86_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/89_83_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/90_85_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/88_85_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/89_84_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/90_86_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/89_85_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/91_85_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/88_86_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/89_87_image.tif',
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/88_83_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/88_84_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/88_87_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/99_55_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/100_56_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/96_55_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/98_53_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/98_54_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/97_55_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/98_55_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/102_56_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/101_56_image.tif',
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/99_56_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/据集/v2/image/96_54_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/97_53_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/97_54_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/96_53_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/97_56_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/98_56_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/96_56_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/9_115_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/5_114_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/9_118_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/9_114_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/8_114_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/5_115_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/8_115_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/11_117_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/10_118_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/11_116_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/11_115_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/10_116_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/5_117_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/9_116_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/6_117_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/6_116_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/4_114_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/7_115_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/10_114_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/9_117_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/11_114_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/10_117_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/7_114_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/6_114_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/8_117_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/7_117_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/4_115_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/8_116_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/10_115_image.tif', 
     '/media/dell/DATA/wy/data/guiyang/数据集/v2/image/7_116_image.tif']
data_path = "/media/dell/DATA/wy/data/guiyang/数据集/v2/"

for path in a:
    index = path.split('/')[-1].split('_')[0]
    for dir, _, files in os.walk(data_path):
        for file in files:
            if file.split('_')[0] == index:
                os.remove(os.path.join(dir, file))