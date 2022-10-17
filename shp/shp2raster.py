import os
import glob
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
def shp2raster(shapename, output_raster, pixel_size):
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

if __name__ == "__main__":
    data_dir = "J:/GuangdongSHP/splitSHP/merge_shp/"
    file_list = glob.glob(('{}*.shp'.format(data_dir)))
    for i, file_name in enumerate(file_list):
        print("{}/{}".format(str(i+1), str(len(file_list))))
        output_raster = file_name.split(".")[0] + '.tif'
        pixel_size = 7.516606439032443e-06
        shp2raster(file_name, output_raster, pixel_size)
