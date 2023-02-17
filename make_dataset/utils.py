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
def get_change_color_table():
    tb = gdal.ColorTable()
    tb.SetColorEntry(0, (0, 0, 0, 255))
    tb.SetColorEntry(1, (255, 255, 255, 255))
    return tb
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