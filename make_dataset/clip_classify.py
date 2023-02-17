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

def choose_best_label_array(label1_files, label2_files, label3_files, extent):
    label1_array = None
    label2_array = None
    label3_array = None
    for i, dataset in enumerate(label1_files):
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
            label1_array = clip_image_array
            label2_array = label2_files[i].ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
            label3_array = label3_files[i].ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
            break
    return label1_array, label2_array, label3_array

def clip_sar_image(sar_dir, img_dir, size, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 获取12个月的sar list
    all_sar_dataset_list = None
    all_sar_list = get_all_type_file(sar_dir, '.tif')
    all_sar_list = sorted(all_sar_list)
    all_sar_dataset_list = [gdal.Open(sar_path) for sar_path in all_sar_list]
    
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
            clip_image = cv2.resize(clip_image.transpose(1, 2, 0), (size[0], size[1]), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
            # 保存裁剪影像
            clip_image_path = os.path.join(each_save_dir, os.path.basename(img_list[j]).replace("image", "sar"))
            clip_image_driver = gdal.GetDriverByName('GTiff')
            clip_image_dataset = clip_image_driver.Create(clip_image_path, size[0], size[1], 2, gdal.GDT_Float32)
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
     
     
    # area = [36527387.25, 36610929.605, 2995506.204, 2906606.027]
    area = [36262215.323, 36356989.263, 2949915.705, 2855935.517]
    pixel_size = [0.65, -0.65]
    size = [1024, 1024]
    
    # 计算两个方向的列表
    num_x = int((area[1] - area[0]) / pixel_size[0] / size[0])
    num_y = int((area[3] - area[2]) / pixel_size[1] / size[1])
    x_list = area[0] + pixel_size[0] * size[0] * (np.arange(0, num_x, 1))
    y_list = area[2] + pixel_size[1] * size[1] * (np.arange(0, num_y, 1))
    
    # 获取影像list
    img_dataset_list = None
    img_dir = "/media/dell/DATA/wy/data/guiyang/合并影像/西秀/2021_nir/"
    img_list = get_all_type_file(img_dir, '.tif')
    img_dataset_list = [gdal.Open(img_path) for img_path in img_list]
    
    # 获取label1 list (非农化)
    label1_dataset_list = None
    label1_dir = "/media/dell/DATA/wy/data/guiyang/标签/分类/西秀/label1/"
    label1_list = get_all_type_file(label1_dir, '.tif')
    label1_dataset_list = [gdal.Open(label_path) for label_path in label1_list]
    # 获取label2 list (非农化+施工)
    label2_dataset_list = None
    label2_dir = "/media/dell/DATA/wy/data/guiyang/标签/分类/西秀/label2/"
    label2_list = get_all_type_file(label2_dir, '.tif')
    label2_dataset_list = [gdal.Open(label_path) for label_path in label2_list]
    # 获取label3 list (非粮化+施工)
    label3_dataset_list = None
    label3_dir = "/media/dell/DATA/wy/data/guiyang/标签/分类/西秀/label3/"
    label3_list = get_all_type_file(label3_dir, '.tif')
    label3_dataset_list = [gdal.Open(label_path) for label_path in label3_list]
    
    all_clipped_num = 0
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            print("x: %d/%d, y: %d/%d" % (i, len(x_list), j, len(y_list)))
            extent = [x, x + pixel_size[0] * size[0], y, y + pixel_size[1] * size[1]]
            # 判断是否存在目标区域的影像
            img_files = find_target_image(img_dataset_list, extent)
            if len(img_files) == 0:
                continue
            label1_files = find_target_image(label1_dataset_list, extent)
            if len(label1_files) == 0:
                continue
            label2_files = find_target_image(label2_dataset_list, extent)
            if len(label2_files) == 0:
                continue
            label3_files = find_target_image(label3_dataset_list, extent)
            if len(label3_files) == 0:
                continue
            
            # 判断目标区域的影像有无nodata/选择最优的影像作为裁剪影像,并得到array
            img_array = choose_best_image_array(img_files, extent)
            if img_array is None or img_array.shape[1] != size[0] or img_array.shape[2] != size[1]:
                continue

            label1, label2, label3 = choose_best_label_array(label1_files, label2_files, label3_files,extent)
            if label1 is None or label2 is None or label3 is None or label1.shape[0] != size[0] or label1.shape[1] != size[1] or label2.shape[0] != size[0] or label2.shape[1] != size[1] or label3.shape[0] != size[0] or label3.shape[1] != size[1]:
                continue
            
            # 保存裁剪后的影像
            
            save_dir = "/media/dell/DATA/wy/data/guiyang/数据集/v2/"
            save_img_path = os.path.join(save_dir, "image", "{}_{}_image.tif".format(i, j))
            if not os.path.exists(os.path.join(save_dir, "image")):
                os.makedirs(os.path.join(save_dir, "image"))
            if not os.path.exists(os.path.join(save_dir, "label1")):
                os.makedirs(os.path.join(save_dir, "label1"))
            if not os.path.exists(os.path.join(save_dir, "label2")):
                os.makedirs(os.path.join(save_dir, "label2"))
            if not os.path.exists(os.path.join(save_dir, "label3")):
                os.makedirs(os.path.join(save_dir, "label3"))
            
            save_label1_path = os.path.join(save_dir, "label1", "{}_{}_label1.tif".format(i, j))
            save_label2_path = os.path.join(save_dir, "label2", "{}_{}_label2.tif".format(i, j))
            save_label3_path = os.path.join(save_dir, "label3", "{}_{}_label3.tif".format(i, j))
            
            
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(save_img_path, size[0], size[1], 4, gdal.GDT_Byte)
            for k in range(4):
                dst_ds.GetRasterBand(k + 1).WriteArray(img_array[k])
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img_dataset_list[0].GetProjection())
            dst_ds.FlushCache()
            dst_ds = None
            
            dst_ds = driver.Create(save_label1_path, size[0], size[1], 1, gdal.GDT_Byte)
            label1[label1==5] = 4
            label1[label1==6] = 5
            label1[label1==7] = 6
            dst_ds.GetRasterBand(1).WriteArray(label1)
            color_table = get_label1_color_table()
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img_dataset_list[0].GetProjection())
            dst_ds.GetRasterBand(1).SetRasterColorTable(color_table)
            dst_ds.FlushCache()
            dst_ds = None
            
            dst_ds = driver.Create(save_label2_path, size[0], size[1], 1, gdal.GDT_Byte)
            label2[label2==5] = 4
            label2[label2==6] = 5
            label2[label2==7] = 6
            label2[label2==11] = 7
            dst_ds.GetRasterBand(1).WriteArray(label2)
            color_table = get_label2_color_table()
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img_dataset_list[0].GetProjection())
            dst_ds.GetRasterBand(1).SetRasterColorTable(color_table)
            dst_ds.FlushCache()
            dst_ds = None
            
            dst_ds = driver.Create(save_label3_path, size[0], size[1], 1, gdal.GDT_Byte)
            label_zeros = np.zeros((size[0], size[1]), dtype=np.uint8)
            label_zeros[label3==8] = 1
            label_zeros[label3==9] = 2
            label_zeros[label3==10] = 3
            label_zeros[label3==1] = 4
            label_zeros[label3==2] = 5
            label_zeros[label3==3] = 6
            label_zeros[label3==4] = 7
            label_zeros[label3==5] = 7
            label_zeros[label3==6] = 8
            label_zeros[label3==7] = 9
            label_zeros[label3==11] = 10
            dst_ds.GetRasterBand(1).WriteArray(label_zeros)
            color_table = get_label3_color_table()
            dst_ds.SetGeoTransform([x, pixel_size[0], 0, y, 0, pixel_size[1]])
            dst_ds.SetProjection(img_dataset_list[0].GetProjection())
            dst_ds.GetRasterBand(1).SetRasterColorTable(color_table)
            dst_ds.FlushCache()
            dst_ds = None
            
            all_clipped_num += 1
            print("finished: ", i * num_y + j + 1, "/", num_x * num_y)
    print("all_clipped_num: ", all_clipped_num)
    
if __name__ == "__main__":
    # main()
    clip_sar_image("/media/dell/DATA/wy/data/guiyang/sar/西秀/2021/","/media/dell/DATA/wy/data/guiyang/数据集/v2/image/", [1024, 1024], "/media/dell/DATA/wy/data/guiyang/数据集/v2/sar/")