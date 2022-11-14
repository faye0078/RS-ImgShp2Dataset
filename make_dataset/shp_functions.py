import os
import glob
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from configs import HIGHVEGE, LOWVEGE

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
    command = "./ogrmerge.py -single -o {}/merged.shp ".format(shp_dir) + files_string
    print(os.popen(command).read())
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
    newField = ogr.FieldDefn('Vege', ogr.OFTInteger)
    layer.CreateField(newField)

    while feature:
        CC = feature.GetField('CC')
        if CC in LOWVEGE:
            feature.SetField('Vege', 0)
        elif CC in HIGHVEGE:
            feature.SetField('Vege', 1)
        else:
            feature.SetField('Vege', 2)
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
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    image_type = "GTiff"
    driver = gdal.GetDriverByName(image_type)
    new_raster = driver.Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
    band = new_raster.GetRasterBand(1)
    no_data_value = 2
    band.SetNoDataValue(no_data_value)

    # TODO: the color map need to be outside
    colormap = np.zeros((4,3), dtype=np.uint8)
    colormap[0] = [0,255,0]
    colormap[1] =  [255, 0, 0]
    colormap[2] =  [153,102,51]

    ct = gdal.ColorTable()
    for i in range(len(colormap)):
        ct.SetColorEntry(i, tuple(colormap[i]))

    band.SetRasterColorTable(ct)
    band.FlushCache()
    gdal.RasterizeLayer(new_raster, [1], shp_layer, options=["Attribute=Vege"])
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(4326)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())
    return

# if __name__ == "__main__":
#     merge_shp()
#     data_dir = "J:/GuangdongSHP/splitSHP/merge_shp/"
#     file_list = glob.glob(('{}*.shp'.format(data_dir)))
#     for i, file_name in enumerate(file_list):
#         print("{}/{}".format(str(i+1), str(len(file_list))))
#         output_raster = file_name.split(".")[0] + '.tif'
#         pixel_size = 7.516606439032443e-06
#         shp2raster(file_name, output_raster, pixel_size)
