from osgeo import gdal
from utils import *
from raster_functions import *
from configs import get_colormap
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

def save_image_array(img_array, img_save_path, save_geo, GetProjection):
    dirver = gdal.GetDriverByName('GTiff')
    if len(img_array.shape) == 3:
        img_save = dirver.Create(img_save_path, img_array.shape[1], img_array.shape[2], img_array.shape[0], gdal.GDT_Byte)
        img_save.SetGeoTransform(save_geo)
        img_save.SetProjection(GetProjection)
        for i in range(img_array.shape[0]):
            img_save.GetRasterBand(i + 1).WriteArray(img_array[i])
        img_save.FlushCache()
        img_save = None
    if len(img_array.shape) == 2:
        img_save = dirver.Create(img_save_path, img_array.shape[0], img_array.shape[1], 1, gdal.GDT_Byte)
        img_save.SetGeoTransform(save_geo)
        img_save.SetProjection(GetProjection)
        img_save.GetRasterBand(1).WriteArray(img_array)
        ct = get_colormap()
        img_save.GetRasterBand(1).SetRasterColorTable(ct)
        img_save.FlushCache()
        img_save = None

def clip_one_by_one(img, label, img_size, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_dataset = gdal.Open(img)
    label_dataset = gdal.Open(label)
    geo = img_dataset.GetGeoTransform()
    extent = [geo[0], geo[0] + geo[1] * img_dataset.RasterXSize, geo[3], geo[3] + geo[5] * img_dataset.RasterYSize]
    x_num = int(((extent[1] - extent[0]) / geo[1]) / img_size[0])
    y_num = int(((extent[3] - extent[2]) / geo[5]) / img_size[1])
    
    base_name = os.path.basename(img).split('.')[0]
    
    for i in range(x_num):
        for j in range(y_num):
            print('正在处理第' + str(i) + '行第' + str(j) + '列')
            x_min = extent[0] + i * img_size[0] * geo[1]
            x_max = extent[0] + (i + 1) * img_size[0] * geo[1]
            y_min = extent[2] + j * img_size[1] * geo[5]
            y_max = extent[2] + (j + 1) * img_size[1] * geo[5]
            save_path = os.path.join(save_dir, str(i) + '_' + str(j) + '.tif')
            img_array = img_dataset.ReadAsArray(i * img_size[0], j * img_size[1], img_size[0], img_size[1])
            if is_exist_zero_pixel_4(img_array.transpose(1, 2, 0), 1000):
                continue
            label_array = label_dataset.ReadAsArray(i * img_size[0], j * img_size[1], img_size[0], img_size[1])
            # save image and label
            img_save_path = os.path.join(save_dir, "image", base_name, base_name + '_' + str(i) + '_' + str(j) + '_img.tif')
            if not os.path.exists(os.path.dirname(img_save_path)):
                os.makedirs(os.path.dirname(img_save_path))
            label_save_path = os.path.join(save_dir, "label", base_name, base_name + '_' + str(i) + '_' + str(j) + '_label.tif')
            if not os.path.exists(os.path.dirname(label_save_path)):
                os.makedirs(os.path.dirname(label_save_path))
            save_geo = (x_min, geo[1], geo[2], y_min, geo[4], geo[5])
            save_image_array(img_array, img_save_path, save_geo, img_dataset.GetProjection())
            save_image_array(label_array, label_save_path, save_geo, label_dataset.GetProjection())

def clip_sar_image(sar_dir, img_dir, size, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 获取12个月的sar list
    all_sar_list = []
    for i in range(12):
        sar_file = os.path.join(sar_dir, str(i + 1) + '.tif')
        all_sar_list.append(sar_file)
    all_sar_dataset_list = [gdal.Open(sar_path) for sar_path in all_sar_list]
    # all_sar_dataset_list = None
    # all_sar_list = get_all_type_file(sar_dir, '.tif')
    # all_sar_list = sorted(all_sar_list)
    # all_sar_dataset_list = [gdal.Open(sar_path) for sar_path in all_sar_list if 'cut' not in sar_path]
    
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
            clip_image_dataset.SetGeoTransform((img_geo[0], sar_geo[1], 0, img_geo[3], 0, sar_geo[5]))
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
                    
def merge_sar_image(sar_dir):
    dir_list = os.listdir(sar_dir)
    for i, dir in enumerate(dir_list):
        all_sar_list = get_all_type_file(os.path.join(sar_dir, dir), '.tif')
        all_sar_list = " ".join(all_sar_list)
        sar_command = "gdal_merge.py -o {} -of GTiff {}".format(os.path.join(sar_dir, dir + '.tif'), all_sar_list)
        os.system(sar_command)
        print("finished {}/{}".format(i + 1, len(dir_list)))
            
    
def main():
    # 需裁减的地理范围，像元大小与裁剪大小
     
     
    # area = [36527387.25, 36610929.605, 2995506.204, 2906606.027]
    # area = [36262215.323, 36356989.263, 2949915.705, 2855935.517]
    area = [36141121.683, 36641978.935, 3122043.685, 2768824.229]
    pixel_size = [0.65, -0.65]
    size = [1024, 1024]
    
    # 计算两个方向的列表
    num_x = int((area[1] - area[0]) / pixel_size[0] / size[0])
    num_y = int((area[3] - area[2]) / pixel_size[1] / size[1])
    x_list = area[0] + pixel_size[0] * size[0] * (np.arange(0, num_x, 1))
    y_list = area[2] + pixel_size[1] * size[1] * (np.arange(0, num_y, 1))
    
    # 获取影像list
    img_dataset_list = None
    img_dir = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/合并影像/全省/2021_nir/"
    img_list = get_all_type_file(img_dir, '.tif')
    img_dataset_list = [gdal.Open(img_path) for img_path in img_list]
    
    # 获取label list
    label_dataset_list = None
    label_dir = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/label/全省-16类/"
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
            if img_array is None:
                continue
            img_array = cv2.resize(img_array.transpose(1, 2, 0), (size[0], size[1]), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
            label = choose_best_label_array(label_files, extent)
            
            if label is None:
                continue
            label = cv2.resize(label, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
            
            # 保存裁剪后的影像
            save_dir = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/数据集/v2/"
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
            color_table = get_colormap()
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img_dataset_list[0].GetProjection())
            dst_ds.GetRasterBand(1).SetRasterColorTable(color_table)
            dst_ds.FlushCache()
            dst_ds = None
        
            all_clipped_num += 1
            print("finished: ", i * num_y + j + 1, "/", num_x * num_y)
    print("all_clipped_num: ", all_clipped_num)
    
def main_clip_one_by_one(img_dir, label_dir, save_dir):
    img_list = get_all_type_file(img_dir, '.tif')
    for img_path in img_list:
        img_name = os.path.basename(img_path).split('.')[0]
        label_path = os.path.join(label_dir, img_name + '.tif')
        if not os.path.exists(label_path):
            print("label not exist: ", label_path)
            exit(0)
        print(img_path)
        print(label_path)
        clip_one_by_one(img_path, label_path, [1024, 1024], save_dir)
    
    
if __name__ == "__main__":
    # main()
    # main_clip_one_by_one("/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/合并影像/全省/2021_nir/", "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/label/全省-16类/", "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/数据集/v3/")
    clip_sar_image("/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/sar/全省/","/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/数据集/v2/image/", [1024, 1024], "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/数据集/v2/sar/")
    # merge_sar_image("/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/guiyang/sar/全省/")