import glob
from osgeo import gdal
from osgeo import ogr

def searchShpByRaster(img_name, shp_dir):
    """search the shp file which is covered by the image

    Args:
        img_name (str): the path of the image
        shp_dir (str): the dir path of all shp files

    Returns:
        list, float: the list of the shp file which is covered by the image, the pixel size of the image
    """    
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

def clipMaskByImg(img_name, mask_name):
    """clip the mask by the image

    Args:
        img_name (str): the path of the image
        mask_name (str): the path of the mask
    """    
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