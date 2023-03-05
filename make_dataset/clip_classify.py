from osgeo import gdal
from utils import *
from raster_functions import *
import os
import numpy as np
import cv2

def is_exist_zero_pixel_4(array, max_num):
    a = np.where(np.all(array == [0, 0, 0, 0], axis=-1))[:2]
    if len(a[0]) > max_num:
        return True
    else:
        return False   
    
def is_exist_zero_pixel_1(array, max_num):
    num = sum(sum(array == 255))
    if num > max_num:
        return True
    else:
        return False
    
def find_target_image(dataset_list, extent):
    target_images = []
    for dataset in dataset_list:
        image_geo = dataset.GetGeoTransform()
        image_extent = [image_geo[0], image_geo[0] + image_geo[1] * dataset.RasterXSize, image_geo[3], image_geo[3] + image_geo[5] * dataset.RasterYSize]
            # 可能需要修改判断条件
        if image_extent[0] <= extent[0] and image_extent[1] >= extent[1] and image_extent[2] >= extent[2] and image_extent[3] <= extent[3]:
            target_images.append(dataset)
    return target_images

def find_target_sar(dataset_dict, extent):
    target_images = OrderedDict()
    for index in dataset_dict:
        dataset = dataset_dict[index]
        image_geo = dataset.GetGeoTransform()
        image_extent = [image_geo[0], image_geo[0] + image_geo[1] * dataset.RasterXSize, image_geo[3], image_geo[3] + image_geo[5] * dataset.RasterYSize]
            # 可能需要修改判断条件
        if image_extent[0] <= extent[0] and image_extent[1] >= extent[1] and image_extent[2] >= extent[2] and image_extent[3] <= extent[3]:
            target_images[index] = dataset
    return target_images


def choose_best_image_array(img_files, extent):
    result_array = None
    for dataset in img_files:
        image_extent = dataset.GetGeoTransform()
        # 计算裁剪范围
        x_min = int((extent[0] - image_extent[0]) / image_extent[1])
        y_min = int((extent[2] - image_extent[3]) / image_extent[5])
        x_max = int((extent[1] - image_extent[0]) / image_extent[1])
        y_max = int((extent[3] - image_extent[3]) / image_extent[5])
        # 裁剪影像
        clip_image_array = dataset.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
        if is_exist_zero_pixel_4(clip_image_array.transpose(1, 2, 0), 5000):
            continue
        else:
            result_array = clip_image_array
            break
    return result_array

def choose_best_label_array(label_files, extent):
    label_array = None
    for i, dataset in enumerate(label_files):
        image_extent = dataset.GetGeoTransform()
        # 计算裁剪范围
        x_min = int((extent[0] - image_extent[0]) / image_extent[1])
        y_min = int((extent[2] - image_extent[3]) / image_extent[5])
        x_max = int((extent[1] - image_extent[0]) / image_extent[1])
        y_max = int((extent[3] - image_extent[3]) / image_extent[5])
        # 裁剪影像
        clip_image_array = dataset.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
        if is_exist_zero_pixel_1(clip_image_array, 0):
            continue
        else:
            label_array = clip_image_array
            break
    return label_array

def clip_sar_image(sar_dir, img_dir, size, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 获取12个月的sar list
    all_sar_dataset_list = None
    all_sar_list = get_all_type_file(sar_dir, '.tif')
    all_sar_list = sorted(all_sar_list)
    all_sar_dataset_list = [gdal.Open(sar_path) for sar_path in all_sar_list if 'cut' not in sar_path]
    
    img_list = get_all_type_file(img_dir, '.tif')
    error_list = []
    for i, sar_dataset in enumerate(all_sar_dataset_list):
        
        each_save_dir = os.path.join(save_dir, str(i + 1))
        if not os.path.exists(each_save_dir):
            os.makedirs(each_save_dir)
        for j, img_path in enumerate(img_list):
            a = j + 1 + i * len(img_list)
            img_dataset = gdal.Open(img_path)
            img_geo = img_dataset.GetGeoTransform()
            sar_geo = sar_dataset.GetGeoTransform()
            img_extent = [img_geo[0], img_geo[0] + img_geo[1] * img_dataset.RasterXSize, img_geo[3], img_geo[3] + img_geo[5] * img_dataset.RasterYSize]
            # 计算裁剪范围
            x_min = int((img_extent[0] - sar_geo[0]) / sar_geo[1])
            y_min = int((img_extent[2] - sar_geo[3]) / sar_geo[5])
            x_max = int((img_extent[1] - sar_geo[0]) / sar_geo[1])
            y_max = int((img_extent[3] - sar_geo[3]) / sar_geo[5])
            # 裁剪影像
            clip_image = sar_dataset.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
            if clip_image is None:
                if img_path not in error_list:
                    error_list.append(img_path)
                continue
            # clip_image = cv2.resize(clip_image.transpose(1, 2, 0), (size[0], size[1]), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
            # 保存裁剪影像
            clip_image_path = os.path.join(each_save_dir, os.path.basename(img_list[j]).replace("image", "sar"))
            clip_image_driver = gdal.GetDriverByName('GTiff')
            # clip_image_dataset = clip_image_driver.Create(clip_image_path, size[0], size[1], 2, gdal.GDT_Float32)
            clip_image_dataset = clip_image_driver.Create(clip_image_path, clip_image.shape[2], clip_image.shape[1], 2, gdal.GDT_Float32)
            clip_image_dataset.SetGeoTransform((img_geo[0], img_geo[1], 0, img_geo[3], 0, img_geo[5]))
            clip_image_dataset.SetProjection(sar_dataset.GetProjection())
            clip_image_dataset.GetRasterBand(1).WriteArray(clip_image[0])
            clip_image_dataset.GetRasterBand(2).WriteArray(clip_image[1])
            clip_image_dataset.FlushCache()
            clip_image_dataset = None
            
            print("finished {}/{}".format(j + 1 + i * len(img_list), len(img_list) * len(all_sar_dataset_list)))
    print(error_list)
    data_path = "/media/dell/DATA/wy/data/guiyang/数据集/v2/"

    for path in error_list:
        index = path.split('/')[-1].split('_')[0]
        for dir, _, files in os.walk(data_path):
            for file in files:
                if file.split('_')[0] == index:
                    os.remove(os.path.join(dir, file))
    
def main():
    # 需裁减的地理范围，像元大小与裁剪大小
     
     
    area = [36527387.25, 36610929.605, 2995506.204, 2906606.027]
    # area = [36262215.323, 36356989.263, 2949915.705, 2855935.517]
    pixel_size = [0.65, -0.65]
    size = [1024, 1024]
    
    # 计算两个方向的列表
    num_x = int((area[1] - area[0]) / pixel_size[0] / size[0])
    num_y = int((area[3] - area[2]) / pixel_size[1] / size[1])
    x_list = area[0] + pixel_size[0] * size[0] * (np.arange(0, num_x, 1))
    y_list = area[2] + pixel_size[1] * size[1] * (np.arange(0, num_y, 1))
    
    # 获取影像list
    img_dataset_list = None
    img_dir = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/合并影像/剑河/2021_nir/"
    img_list = get_all_type_file(img_dir, '.tif')
    img_dataset_list = [gdal.Open(img_path) for img_path in img_list]
    
    # 获取label list
    label_dataset_list = None
    label_dir = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/label/剑河/"
    label_list = get_all_type_file(label_dir, '.tif')
    label_dataset_list = [gdal.Open(label_path) for label_path in label_list]
    
    all_clipped_num = 0
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            print("x: %d/%d, y: %d/%d" % (i, len(x_list), j, len(y_list)))
            extent = [x, x + pixel_size[0] * size[0], y, y + pixel_size[1] * size[1]]
            # 判断是否存在目标区域的影像
            img_files = find_target_image(img_dataset_list, extent)
            if len(img_files) == 0:
                continue
            label_files = find_target_image(label_dataset_list, extent)
            if len(label_files) == 0:
                continue
            
            # 判断目标区域的影像有无nodata/选择最优的影像作为裁剪影像,并得到array
            img_array = choose_best_image_array(img_files, extent)
            if img_array is None or img_array.shape[1] != size[0] or img_array.shape[2] != size[1]:
                continue

            label = choose_best_label_array(label_files, extent)
            if label is None or label.shape[0] != size[0] or label.shape[1] != size[1]:
                continue
            
            # 保存裁剪后的影像
            save_dir = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/数据集/v1/"
            save_img_path = os.path.join(save_dir, "image", "{}_{}_image.tif".format(i, j))
            if not os.path.exists(os.path.join(save_dir, "image")):
                os.makedirs(os.path.join(save_dir, "image"))
            if not os.path.exists(os.path.join(save_dir, "label")):
                os.makedirs(os.path.join(save_dir, "label"))
            
            save_label_path = os.path.join(save_dir, "label", "{}_{}_label.tif".format(i, j))
            
            
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(save_img_path, size[0], size[1], 4, gdal.GDT_Byte)
            for k in range(4):
                dst_ds.GetRasterBand(k + 1).WriteArray(img_array[k])
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img_dataset_list[0].GetProjection())
            dst_ds.FlushCache()
            dst_ds = None
            
            dst_ds = driver.Create(save_label_path, size[0], size[1], 1, gdal.GDT_Byte)
            dst_ds.GetRasterBand(1).WriteArray(label)
            color_table = get_label1_color_table()
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img_dataset_list[0].GetProjection())
            dst_ds.GetRasterBand(1).SetRasterColorTable(color_table)
            dst_ds.FlushCache()
            dst_ds = None
        
            all_clipped_num += 1
            print("finished: ", i * num_y + j + 1, "/", num_x * num_y)
    print("all_clipped_num: ", all_clipped_num)
    
if __name__ == "__main__":
    main()
    # clip_sar_image("/media/dell/DATA/wy/data/guiyang/sar/剑河/2021/","/media/dell/DATA/wy/data/guiyang/数据集/v3/分类/剑河/image/", [1024, 1024], "/media/dell/DATA/wy/data/guiyang/数据集/v3/分类/剑河/sar/")