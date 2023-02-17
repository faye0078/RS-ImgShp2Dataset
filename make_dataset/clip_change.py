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
    num = sum(sum(array == 0))
    num_15 = sum(sum(array == 15))
    if num > max_num or num_15 > max_num:
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
        if is_exist_zero_pixel_1(clip_image_array, 100):
            continue
        else:
            label_array = clip_image_array
            break
    return label_array

def read_label_array(label_dataset, extent):
    image_extent = label_dataset.GetGeoTransform()
    # 计算裁剪范围
    x_min = int((extent[0] - image_extent[0]) / image_extent[1])
    y_min = int((extent[2] - image_extent[3]) / image_extent[5])
    x_max = int((extent[1] - image_extent[0]) / image_extent[1])
    y_max = int((extent[3] - image_extent[3]) / image_extent[5])
    # 裁剪影像
    clip_image_array = label_dataset.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
    return clip_image_array
    
def main():
    # 需裁减的地理范围，像元大小与裁剪大小
    # area = [36527387.25, 36610929.605, 2995506.204, 2906606.027]
    area = [36262215.323, 36356989.263, 2949915.705, 2855935.517]
    pixel_size = [0.65, -0.65]
    size = [1024, 1024]
    
    # 计算两个方向的列表
    num_x = int((area[1] - area[0]) / pixel_size[0] / size[0])
    num_y = int((area[3] - area[2]) / pixel_size[1] / size[1])
    x_list = area[0] + pixel_size[0] * size[0] * (np.arange(0, num_x, 1))
    y_list = area[2] + pixel_size[1] * size[1] * (np.arange(0, num_y, 1))
    
    # 获取2020影像list
    img1_dataset_list = None
    img1_dir = "/media/dell/DATA/wy/data/guiyang/合并影像/西秀/2020_nir/"
    img1_list = get_all_type_file(img1_dir, '.tif')
    img1_dataset_list = [gdal.Open(img_path) for img_path in img1_list]
    
    # 获取2021影像list
    img2_dataset_list = None
    img2_dir = "/media/dell/DATA/wy/data/guiyang/合并影像/西秀/2021_nir/"
    img2_list = get_all_type_file(img2_dir, '.tif')
    img2_dataset_list = [gdal.Open(img_path) for img_path in img2_list]
    
    # 获取label list (非农化)用于判断是否没有标签
    label_dataset_list = None
    label_dir = "/media/dell/DATA/wy/data/guiyang/标签/分类/西秀/label1/"
    label_list = get_all_type_file(label_dir, '.tif')
    label_dataset_list = [gdal.Open(label_path) for label_path in label_list]
    
    # 获取 2020 label
    label1_path = "/media/dell/DATA/wy/data/guiyang/标签/变化检测/西秀/2020标签/label_2020_trans.tif"
    label1_dataset = gdal.Open(label1_path)
    # 获取 2021 label
    label2_path = "/media/dell/DATA/wy/data/guiyang/标签/变化检测/西秀/2021标签/label_2021_trans.tif"
    label2_dataset = gdal.Open(label2_path)
    
    all_clipped_num = 0
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            print("x: %d/%d, y: %d/%d" % (i, len(x_list), j, len(y_list)))
            extent = [x, x + pixel_size[0] * size[0], y, y + pixel_size[1] * size[1]]
            # 判断是否存在目标区域的影像
            img1_files = find_target_image(img1_dataset_list, extent)
            if len(img1_files) == 0:
                continue
            img2_files = find_target_image(img2_dataset_list, extent)
            if len(img2_files) == 0:
                continue
            label1_files = find_target_image([label1_dataset], extent)
            label2_files = find_target_image([label2_dataset], extent)
            if len(label1_files) == 0 or len(label2_files) == 0:
                continue
            label_files = find_target_image(label_dataset_list, extent)
            if len(label_files) == 0:
                continue
            label = choose_best_label_array(label_files, extent)
            if label is None:
                continue
            # 判断目标区域的影像有无nodata/选择最优的影像作为裁剪影像,并得到array
            img1_array = choose_best_image_array(img1_files, extent)
            img2_array = choose_best_image_array(img2_files, extent)
            if img1_array is None or img1_array.shape[1] != size[0] or img1_array.shape[2] != size[1]:
                continue
            if img2_array is None or img2_array.shape[1] != size[0] or img2_array.shape[2] != size[1]:
                continue
            
            label1_array = read_label_array(label1_dataset, extent)
            label2_array = read_label_array(label2_dataset, extent)
            if label1_array.shape[0] != size[0] or label1_array.shape[1] != size[1]:
                label1_array = cv2.resize(label1_array, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
            if label2_array.shape[0] != size[0] or label2_array.shape[1] != size[1]:
                label2_array = cv2.resize(label2_array, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
            # 保存裁剪后的影像
            
            save_dir = "/media/dell/DATA/wy/data/guiyang/数据集/v2/变化检测/西秀/"
            save_img1_path = os.path.join(save_dir, "image1", "{}_{}_image1.tif".format(i, j))
            if not os.path.exists(os.path.join(save_dir, "image1")):
                os.makedirs(os.path.join(save_dir, "image1"))
            if not os.path.exists(os.path.join(save_dir, "image2")):
                os.makedirs(os.path.join(save_dir, "image2"))
            if not os.path.exists(os.path.join(save_dir, "label1")):
                os.makedirs(os.path.join(save_dir, "label1"))
            if not os.path.exists(os.path.join(save_dir, "label2")):
                os.makedirs(os.path.join(save_dir, "label2"))
            
            save_img2_path = os.path.join(save_dir, "image2", "{}_{}_image2.tif".format(i, j))
            save_label1_path = os.path.join(save_dir, "label1", "{}_{}_label1.tif".format(i, j))
            save_label2_path = os.path.join(save_dir, "label2", "{}_{}_label2.tif".format(i, j))
            
            
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(save_img1_path, size[0], size[1], 4, gdal.GDT_Byte)
            for k in range(4):
                dst_ds.GetRasterBand(k + 1).WriteArray(img1_array[k])
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img1_dataset_list[0].GetProjection())
            dst_ds.FlushCache()
            dst_ds = None
            
            dst_ds = driver.Create(save_img2_path, size[0], size[1], 4, gdal.GDT_Byte)
            for k in range(4):
                dst_ds.GetRasterBand(k + 1).WriteArray(img2_array[k])
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img2_dataset_list[0].GetProjection())
            dst_ds.FlushCache()
            dst_ds = None
            
            dst_ds = driver.Create(save_label1_path, size[0], size[1], 1, gdal.GDT_Byte)
            dst_ds.GetRasterBand(1).WriteArray(label1_array)
            color_table = get_change_color_table()
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img1_dataset_list[0].GetProjection())
            dst_ds.GetRasterBand(1).SetRasterColorTable(color_table)
            dst_ds.FlushCache()
            dst_ds = None
            
            dst_ds = driver.Create(save_label2_path, size[0], size[1], 1, gdal.GDT_Byte)
            dst_ds.GetRasterBand(1).WriteArray(label2_array)
            color_table = get_change_color_table()
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img1_dataset_list[0].GetProjection())
            dst_ds.GetRasterBand(1).SetRasterColorTable(color_table)
            dst_ds.FlushCache()
            dst_ds = None
            
            all_clipped_num += 1
            print("finished: ", i * num_y + j + 1, "/", num_x * num_y)
    print("all_clipped_num: ", all_clipped_num)
    
if __name__ == "__main__":
    main()