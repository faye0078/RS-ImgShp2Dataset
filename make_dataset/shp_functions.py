import os
import glob
import numpy as np
import time
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from configs import *

def merge_shp(shp_list, save_dir):
    """merge shapefiles in shp_list to a single shapefile in save_dir

    Args:
        shp_list (list): _description_
        save_dir (str): the path of save dir

    Returns:
        str: the merged shp file path
    """    
    files_string = " ".join(shp_list)
    print(files_string)
    shp_dir = os.path.join(os.path.dirname(shp_list[0]), save_dir)
    if not os.path.exists(shp_dir):
        os.makedirs(shp_dir)
    
    # the path maybe need to be changed
    command = "ogrmerge.py -single -o {}/merged.shp ".format(shp_dir) + files_string
    print(os.popen(command).read())
    time.sleep(1)
    return shp_dir + "/merged.shp"

def trans_shp(fn):
    """create a new feature depending on the 'CC' field

    Args:
        fn (function): _description_
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(fn, 1)
    layer = dataSource.GetLayer()
    feature = layer.GetNextFeature()
    sum = 0
    newField = ogr.FieldDefn('My_class', ogr.OFTInteger)
    if layer.GetLayerDefn().GetFieldIndex('My_class') == -1:
        layer.CreateField(newField)
    while feature:
        DLBM = feature.GetField('DLBM')
        # if DLBM in 水田:
        #     feature.SetField('My_class', 0)
        # elif DLBM in 旱地:
        #     feature.SetField('My_class', 1)
        # elif DLBM in 果园:
        #     feature.SetField('My_class', 2)
        # elif DLBM in 茶园:
        #     feature.SetField('My_class', 3)
        # elif DLBM in 乔木林地:
        #     feature.SetField('My_class', 4)
        # elif DLBM in 灌木林地:
        #     feature.SetField('My_class', 5)
        # elif DLBM in 苗圃:
        #     feature.SetField('My_class', 6)
        # elif DLBM in 草地:
        #     feature.SetField('My_class', 7)
        # elif DLBM in 工矿用地:
        #     feature.SetField('My_class', 8)
        # elif DLBM in 公共建筑:
        #     feature.SetField('My_class', 9)
        # elif DLBM in 城镇住宅:
        #     feature.SetField('My_class', 10)
        # elif DLBM in 农村住宅:
        #     feature.SetField('My_class', 11)
        # elif DLBM in 公路用地:
        #     feature.SetField('My_class', 12)
        # elif DLBM in 农村道路: 
        #     feature.SetField('My_class', 13)
        # elif DLBM in 河流:
        #     feature.SetField('My_class', 14)
        # elif DLBM in 裸地:
        #     feature.SetField('My_class', 15)
        # else:
        #     feature.SetField('My_class', 16)
        #     sum += 1
        if DLBM in 田地:
            feature.SetField('My_class', 0)
        elif DLBM in 园地:
            feature.SetField('My_class', 1)
        elif DLBM in 林地:
            feature.SetField('My_class', 2)
        elif DLBM in 建筑用地:
            feature.SetField('My_class', 3)
        elif DLBM in 道路:
            feature.SetField('My_class', 4)
        elif DLBM in 水体:
            feature.SetField('My_class', 5)
        else:
            feature.SetField('My_class', 6)
            sum += 1
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    print(sum)
    return

def trans_shp_all_class(fn):
    """create a new feature depending on the 'CC' field

    Args:
        fn (function): _description_
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(fn, 1)
    layer = dataSource.GetLayer()
    feature = layer.GetNextFeature()
    newField = ogr.FieldDefn('My_class', ogr.OFTInteger)
    if layer.GetLayerDefn().GetFieldIndex('My_class') == -1:
        layer.CreateField(newField)
    while feature:
        DLBM = feature.GetField('DLBM')
        if DLBM not in CORRESPOND:
            code = 56
        else:
            code = CORRESPOND_LABEL[CORRESPOND[DLBM]]

        feature.SetField('My_class', code)
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    return

def shp2raster(shapename, output_raster, pixel_size, colormap=None):
    """convert shapefile to raster

    Args:
        shapename (str): the path of shapefile
        output_raster (str): the path of output raster
        pixel_size (float): the pixel size of output raster
        colormap(array): the color map of output raster
    """    
    input_shp = ogr.Open(shapename)
    shp_layer = input_shp.GetLayer()
    extent = shp_layer.GetExtent()
    x_min = extent[0]
    x_max = extent[1]
    y_min = extent[2]
    y_max = extent[3]

    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    image_type = "GTiff"
    driver = gdal.GetDriverByName(image_type)
    new_raster = driver.Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = new_raster.GetRasterBand(1)
    ct = colormap

    # band.SetRasterColorTable(ct)
    band.SetNoDataValue(255)
    band.FlushCache()
    gdal.RasterizeLayer(new_raster, [1], shp_layer, options=["Attribute=My_class"])
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(4524)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())
    return

def count_features_by_field(shp_file, field_name):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(shp_file, 0)
    layer = data_source.GetLayer()
    feature_count = {}
    for feature in layer:
        field_value = feature.GetField(field_name)
        if field_value not in feature_count:
            feature_count[field_value] = 1
        else:
            feature_count[field_value] += 1
    return feature_count

def area_features_by_field(shp_file):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(shp_file, 0)
    layer = data_source.GetLayer()
    feature_area = {}
    for feature in layer:
        field_value = feature.GetField("DLBM")
        field_area = feature.GetField("SHAPE_Area")
        if field_value not in feature_area:
            feature_area[field_value] = field_area
        else:
            feature_area[field_value] += field_area
    return feature_area

def gdb_to_shp(gdb_file, output_folder):
    ogr_command = "ogr2ogr -f 'ESRI Shapefile' -lco ENCODING=UTF-8 -s_srs EPSG:4490 -t_srs EPSG:4524 {} {}".format(output_folder, gdb_file)
    os.system(ogr_command)
    
def rename_lcpa_copy(shp_dir, target_dir):
    for dir, _, file_names in os.walk(shp_dir):
        for file_name in file_names:
            if "LCPA" in file_name:
                source_file = os.path.join(dir, file_name)
                taget_name = file_name.replace("LCPA", dir.split('/')[-1])
                target_file = os.path.join(target_dir, taget_name)
                os.popen('cp {} {}'.format(source_file, target_file))
    
if __name__ == "__main__":
    a=0
    # gdb_dir = "/media/dell/DATA/wy/data/guiyang/地理国情监测/2021/分区/"
    # output_dir = "/media/dell/DATA/wy/data/guiyang/地理国情监测/2021/shape/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # gdb_list = os.listdir(gdb_dir)
    # for gdb_name in gdb_list:
    #     print(gdb_name)
    #     gdb_path = os.path.join(gdb_dir, gdb_name)
    #     output_shp_dir = os.path.join(output_dir, gdb_name.split('.')[0])
    #     if not os.path.exists(output_shp_dir):
    #         os.makedirs(output_shp_dir)
    #     gdb_to_shp(gdb_path, output_shp_dir)
    # rename_lcpa_copy("/media/dell/DATA/wy/data/guiyang/地理国情监测/2021/shape/", "/media/dell/DATA/wy/data/guiyang/地理国情监测/2021/LCPA/")
#     merge_shp()
#     data_dir = "J:/GuangdongSHP/splitSHP/merge_shp/"
#     file_list = glob.glob(('{}*.shp'.format(data_dir)))
#     for i, file_name in enumerate(file_list):
#         print("{}/{}".format(str(i+1), str(len(file_list))))
#         output_raster = file_name.split(".")[0] + '.tif'
#         pixel_size = 7.516606439032443e-06
#         shp2raster(file_name, output_raster, pixel_size)
