import glob
from osgeo import gdal
from osgeo import ogr, osr
from PIL import Image
import os 
import numpy as np
import tifffile as tiff
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

def split_img_label(img, label, label3, sar_name_list, split_size, overlap_size, save_path):
    """split the image and the label

    Args:
        img_path (str): image path
        label_path (str): label path
        split_size (int): the size of the split image
        overlap_size (int): the size of the overlap
        save_path (str): the path of the save image and label
    """    

    img_name = os.path.basename(img)
    label_name = os.path.basename(label)
    label3_name = os.path.basename(label3)
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
    
    all_label_array = label_dataset.ReadAsArray(0, 0, img_width, img_height)
    # all_img_array = img_dataset.ReadAsArray(0, 0, img_width, img_height)
    # split the image
    for i in range(split_num_x):
        for j in range(split_num_y):
            print("begin {}/{}".format(i * split_num_y + j, split_num_x * split_num_y))
            left_num = i * (split_size - overlap_size)
            bottom_num = j * (split_size - overlap_size)
            windows = [left_num, bottom_num, split_size, split_size]
            
            label_array = all_label_array[bottom_num:bottom_num+split_size, left_num:left_num+split_size]
            if np.sum(label_array) == 0:
                continue
            # img_array = all_img_array[:, bottom_num:bottom_num+split_size, left_num:left_num+split_size]
            # if np.sum(img_array) == 0:
            #     continue
            
            output_img_name = img_name.replace(".png", "_{}_{}.tif".format(i, j))
            output_label_name = label_name.replace(".tif", "_{}_{}.tif".format(i, j))
            output_label3_name = label3_name.replace(".tif", "_{}_{}.tif".format(i, j))
            
            img_clip_dir = os.path.join(save_path, 'img_block')
            if not os.path.exists(img_clip_dir):
                os.makedirs(img_clip_dir)
            img_clip_command = "gdal_translate -of GTiff -srcwin {} {} {} {} {} {}".format(left_num, bottom_num, split_size, split_size, img, os.path.join(img_clip_dir, output_img_name))
            print(os.popen(img_clip_command).read())
            
            label_clip_dir = os.path.join(save_path, 'label_block')
            if not os.path.exists(label_clip_dir):
                os.makedirs(label_clip_dir)
            label_clip_command = "gdal_translate -of GTiff -srcwin {} {} {} {} {} {}".format(left_num, bottom_num, split_size, split_size, label, os.path.join(label_clip_dir, output_label_name))
            print(os.popen(label_clip_command).read())
            
            label3_clip_dir = os.path.join(save_path, 'label3_block')
            if not os.path.exists(label3_clip_dir):
                os.makedirs(label3_clip_dir)
            label3_clip_command = "gdal_translate -of GTiff -srcwin {} {} {} {} {} {}".format(left_num, bottom_num, split_size, split_size, label3, os.path.join(label3_clip_dir, output_label3_name))
            print(os.popen(label3_clip_command).read())
            
            # gdal.Translate(os.path.join(save_path, 'img_clip', output_img_name), img_dataset, srcWin=windows)
            # gdal.Translate(os.path.join(save_path, 'label_clip', output_label_name), label_dataset, srcWin=windows)
            sar_clip_dir = os.path.join(save_path, 'sar_block')
            if not os.path.exists(sar_clip_dir):
                os.makedirs(sar_clip_dir)
            for k in range(len(sar_dataset_list)):
                sar_name = os.path.basename(sar_name_list[k])
                output_sar_name = sar_name.replace(".tif", "_{}_{}.tif".format(i, j))
                sar_clip_command = "gdal_translate -of GTiff -srcwin {} {} {} {} {} {}".format(left_num, bottom_num, split_size, split_size, sar_name_list[k], os.path.join(sar_clip_dir, output_sar_name))
                print(os.popen(sar_clip_command).read())
                # gdal.Translate(os.path.join(save_path, 'sar_clip', output_sar_name), sar_dataset_list[k], srcWin=windows)
                
            print("finish {}/{}".format(i * split_num_y + j, split_num_x * split_num_y))

    
    return save_path
    

def add_builtup_label(label_path, builtup_label_path, save_path):
    """add the builtup label to the label

    Args:
        label_path (str): label dir
        builtup_label_path (str): builtup label path
    """    
    
    label_dataset = gdal.Open(label_path)
    builtup_dataset = gdal.Open(builtup_label_path)

    label_band = label_dataset.GetRasterBand(1)
    label_data = label_band.ReadAsArray()
    builtup_band = builtup_dataset.GetRasterBand(1)
    builtup_data = builtup_band.ReadAsArray()
    
    if label_data.shape != builtup_data.shape:
        print('the shape of the label and the builtup label is not the same')
        return
    label_data[label_data==127] = 0
    label_data[builtup_data == 1] = 11
    driver = gdal.GetDriverByName("GTiff")
    img_name = os.path.basename(label_path).replace(".png", ".tif")
    result_name = os.path.join(save_path, img_name)
    dst_ds = driver.Create(result_name, label_dataset.RasterXSize, label_dataset.RasterYSize, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(label_dataset.GetGeoTransform())
    dst_ds.SetProjection(label_dataset.GetProjection())
    dst_ds.GetRasterBand(1).WriteArray(label_data)
    dst_ds.FlushCache()
    dst_ds = None
    
    return result_name

def clip_builtup(img_path, builtup_path, save_path):
    if not os.path.exists(builtup_path):
        builtup_path = builtup_path.replace(".tif", ".png")
    img_dataset = gdal.Open(img_path)
    img_geo = img_dataset.GetGeoTransform()
    img_extent = [img_geo[0], img_geo[0] + img_geo[1] * img_dataset.RasterXSize, img_geo[3] + img_geo[5] * img_dataset.RasterYSize, img_geo[3]]
    
    img_name = os.path.basename(img_path).replace(".png", ".tif")
    clip_path = os.path.join(save_path, img_name)
    clip_command = "gdal_translate -projwin {} {} {} {} -of GTiff {} {}".format(img_extent[0], img_extent[3], img_extent[1], img_extent[2], builtup_path, clip_path)
    print(clip_command)
    print(os.popen(clip_command).read())
    print("builtup clip finished: ", img_name)
    
    return clip_path

def clip_label(img_path, label_path, save_path):
    if not os.path.exists(label_path):
        label_path = label_path.replace(".tif", ".png")
    img_dataset = gdal.Open(img_path)
    img_geo = img_dataset.GetGeoTransform()
    img_extent = [img_geo[0], img_geo[0] + img_geo[1] * img_dataset.RasterXSize, img_geo[3] + img_geo[5] * img_dataset.RasterYSize, img_geo[3]]
    
    img_name = os.path.basename(img_path).replace(".png", ".tif")
    clip_path = os.path.join(save_path, img_name)
    clip_command = "gdal_translate -projwin {} {} {} {} -of GTiff {} {}".format(img_extent[0], img_extent[3], img_extent[1], img_extent[2], label_path, clip_path)
    print(os.popen(clip_command).read())
    print("label clip finished: ", img_name)
    
    return clip_path

def clip_sar(img_path, sar_path, save_path):
    img_dataset = gdal.Open(img_path)
    img_geo = img_dataset.GetGeoTransform()
    img_extent = [img_geo[0], img_geo[0] + img_geo[1] * img_dataset.RasterXSize, img_geo[3] + img_geo[5] * img_dataset.RasterYSize, img_geo[3]]
    mouth = sar_path.split("/")[-2]
    
    img_name = os.path.basename(img_path).replace(".png", ".tif")
    resample_path = os.path.join(save_path, mouth + "_resample_" + img_name)
    resample_command = "gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4524 -tr 0.65 0.65 -r near -of GTiff {} {}".format(sar_path, resample_path)
    print(os.popen(resample_command).read())
    
    clip_path = os.path.join(save_path, mouth + "_" + img_name)
    clip_command = "gdal_translate -projwin {} {} {} {} -of GTiff {} {}".format(img_extent[0], img_extent[3], img_extent[1], img_extent[2], resample_path, clip_path)
    print(os.popen(clip_command).read())
    print("sar clip finished: ", img_name)
    
    return clip_path

def label_tif2png(img_path):
    img_dataset = gdal.Open(img_path)
    img_data = img_dataset.GetRasterBand(1).ReadAsArray()
    img_data = img_data.astype(np.uint8)
    img_data[img_data == 255] = 0
    img_data[img_data == 1] = 255
    img_data = Image.fromarray(img_data)
    img_name = img_path.replace("conslabel.tif", "label.png")
    img_data.save(img_name)
    print("tif2png finished: ", img_name)
    
    return img_name

def img_save(img_path):
    img_dataset = gdal.Open(img_path)
    img_data = img_dataset.ReadAsArray()
    img_data = np.transpose(img_data, (1, 2, 0))
    img_data = img_data.astype(np.uint16)
    img_name = img_path.replace("img", "image")
    tiff.imwrite(img_name, img_data)
    print("image save finished: ", img_name)
    
    return img_name

def png2tif(img_path):
    output_path = img_path.replace("png", "tif")
    trans_command = "gdal_translate -a_srs EPSG:4524 -of GTiff {} {}".format(img_path, output_path)
    print(os.popen(trans_command).read())
    
def gdal_merge_multi(tif_dir):
    tif_list = glob.glob(os.path.join(tif_dir, "*.tif"))
    tif_list = " ".join(tif_list)
    merge_command = "gdal_merge.py -o {} {}".format(os.path.join(tif_dir, "merge.tif"), tif_list)
    os.system(merge_command)
    
def gdal_swarp_to_4524(tif_path, result_path):
    swarp_command = "gdalwarp -t_srs EPSG:4524 -tr 0.65 0.65 -of GTiff {} {}".format(tif_path, result_path)
    os.system(swarp_command)