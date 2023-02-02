import glob
from osgeo import gdal
from osgeo import ogr
import os 
def search_shp_by_raster(img_name, shp_dir):
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

def clip_mask_by_img(img_name, mask_name):
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
    
def check_geo_params(img_path, label_path):
    """check the geo params of the image and the label

    Args:
        img_path (str): image base path
        label_path (str): label base path

    Returns:
        right_img_list: the list of the right image
        right_label_list: the list of the right label
    """    
    img_list = glob.glob(('{}*.tif'.format(img_path)))
    right_img_list = []
    right_label_list = []
    
    for img in img_list:
        img_name = os.path.basename(img)
        label_name = img_name.replace(".tif", "_label.tif") # TODO: 对应关系确定

        label_path = os.path.join(label_path, label_name)
        
        img_dataset = gdal.Open(img)
        label_dataset = gdal.Open(label_path)
        img_geo = img_dataset.GetGeoTransform()
        label_geo = label_dataset.GetGeoTransform()
        
        # check the pixel size of the image and the label
        if img_geo[1] != label_geo[1] or img_geo[5] != label_geo[5]:
            print("The pixel size of the {} image and the its label is not equal".format(img_name))
            continue
        
        # check the coordinate of the image and the label
        if img_geo[0] != label_geo[0] or img_geo[3] != label_geo[3]: # TODO: 是否需要做差值判断
            "The coordinate of the {} image and the its label is not equal".format(img_name)
            continue
        
        right_img_list.append(img)
        right_label_list.append(label_path)
        
    return right_img_list, right_label_list

def trans_raster_label(label_path, label_map):
    """transform the raster label to the label which is defined by the user

    Args:
        label_path (str): label path
        label_map (array): label map

    Returns:
        str: the result label path
    """    
    label_list = glob.glob(('{}*.tif'.format(label_path)))
    for label in label_list:
        dataset = gdal.Open(label)
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        for key, value in label_map.items():
            data[data == key] = value
        driver = gdal.GetDriverByName("GTiff")
        
        label_name = os.path.basename(label)
        trans_label_name = label_name.replace(".tif", "_trans_label.tif")
        trans_label_name = os.path.join(label_path, 'tans_label', label_name)
        dst_ds = driver.Create(trans_label_name, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Byte)
        dst_ds.SetGeoTransform(dataset.GetGeoTransform())
        dst_ds.SetProjection(dataset.GetProjection())
        dst_ds.GetRasterBand(1).WriteArray(data)
        dst_ds.FlushCache()
        dst_ds = None
        
    return os.path.join(label_path, 'tans_label')

def split_img_label(img_list, label_list, split_size, overlap_size, sar_path, save_path):
    """split the image and the label

    Args:
        img_path (str): image path
        label_path (str): label path
        split_size (int): the size of the split image
        overlap_size (int): the size of the overlap
        save_path (str): the path of the save image and label
    """    
    for img, label in zip(img_list, label_list):
        
        # get the 12 sar image names
        sar_path_list = os.listdir(sar_path)
        sar_name_list = []
        for sar_path in sar_path_list:
            sar_name = os.path.join(sar_path, os.path.basename(img))
            sar_name_list.append(sar_name)
            
        img_name = os.path.basename(img)
        label_name = os.path.basename(label)
        img_dataset = gdal.Open(img)
        label_dataset = gdal.Open(label)
        sar_dataset_list = []
        for sar_name in sar_name_list:
            sar_dataset = gdal.Open(sar_name)
            sar_dataset_list.append(sar_dataset)
            
        img_width = img_dataset.RasterXSize
        img_height = img_dataset.RasterYSize
        
        # calculate the split number
        split_num_x = int((img_width - overlap_size) / (split_size - overlap_size))
        split_num_y = int((img_height - overlap_size) / (split_size - overlap_size))
        
        # split the image
        for i in range(split_num_x):
            for j in range(split_num_y):
                left_num = i * (split_size - overlap_size)
                bottom_num = j * (split_size - overlap_size)
                windows = [left_num, bottom_num, split_size, split_size]
                gdal.Translate(os.path.join(save_path, 'img', '{}_{}_{}'.format(i, j, img_name)), img_dataset, srcWin=windows)
                gdal.Translate(os.path.join(save_path, 'label', '{}_{}_{}'.format(i, j, label_name)), label_dataset, srcWin=windows)
                for k in range(len(sar_dataset_list)):
                    gdal.Translate(os.path.join(save_path, 'sar', '{}_{}_{}'.format(i, j, sar_name_list[k])), sar_dataset_list[k], srcWin=windows)
                    
        # clear the memory
        img_dataset.FlushCache()
        img_dataset = None
        label_dataset.FlushCache()
        label_dataset = None
        for sar_dataset in sar_dataset_list:
            sar_dataset.FlushCache()
            sar_dataset = None
        
        return save_path