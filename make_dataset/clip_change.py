from osgeo import gdal
import os
import numpy as np

def find_target_image(dir, extent, file_type):
    """find target image acording to extent

    Args:
        dir (str): image directory
        extent (list): extent of target area
        file_type (str): image file type

    Returns:
        list: all target image path
    """    
    target_images = []
    for root, dirs, files in os.walk(dir):
        if len(files) == 0:
            continue
        for file in files:
            if file.endswith(file_type):
                image_path = os.path.join(root, file)
                image = gdal.Open(image_path)
                image_extent = image.GetGeoTransform()
                # 可能需要修改判断条件
                if image_extent[0] <= extent[0] and image_extent[3] >= extent[3] and image_extent[1] >= extent[1] and image_extent[5] <= extent[5]:
                    target_images.append(image_path)
    return target_images
    
def clip_image(image_path, output_path, extent):
    """clip image acording to extent

    Args:
        image_path (str): image path
        output_path (str): output path
        extent (list): extent of target area
    """    
    image = gdal.Open(image_path)
    image_extent = image.GetGeoTransform()
    # 计算裁剪范围
    x_min = int((extent[0] - image_extent[0]) / image_extent[1])
    y_min = int((extent[3] - image_extent[3]) / image_extent[5])
    x_max = int((extent[1] - image_extent[0]) / image_extent[1])
    y_max = int((extent[5] - image_extent[3]) / image_extent[5])
    # 裁剪影像
    clip_image = image.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
    # 保存裁剪影像
    clip_image_path = output_path
    clip_image_driver = gdal.GetDriverByName('GTiff')
    clip_image_dataset = clip_image_driver.Create(clip_image_path, x_max - x_min, y_max - y_min, 1, gdal.GDT_Float32)
    clip_image_dataset.SetGeoTransform((extent[0], image_extent[1], 0, extent[3], 0, image_extent[5]))
    clip_image_dataset.SetProjection(image.GetProjection())
    clip_image_dataset.GetRasterBand(1).WriteArray(clip_image)
    clip_image_dataset.FlushCache()
    clip_image_dataset = None

def clip_change_image(time1_dir, time2_dir, extent):
    """clip two time images acording to extent

    Args:
        time1_dir (str): time one image directory
        time2_dir (str): time two image directory
        extent (list): extent of target area
    """ 
    # 查找目标影像
    time1_images = find_target_image(time1_dir, extent, '.tif')
    time2_images = find_target_image(time2_dir, extent, '.tif')
    
    # 判断是否存在目标影像
    if len(time1_images) == 0 or len(time2_images) == 0:
        return False
    
    # 裁剪时相1影像
    for i, image_path in enumerate(time1_images):
        extent_name = "%.6f_%.6f_%.6f_%.6f" % (extent[0], extent[1], extent[2], extent[3]) #保留6位小数
        output_dir = os.path.join(os.path.dirname(image_path), "time1_clip", extent_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        clip_image(image_path, os.path.join(output_dir, os.path.basename(image_path)), extent)
        
    # 裁剪时相2影像
    for i, image_path in enumerate(time2_images):
        extent_name = "%.6f_%.6f_%.6f_%.6f" % (extent[0], extent[1], extent[2], extent[3]) #保留6位小数
        output_dir = os.path.join(os.path.dirname(image_path), "time2_clip", extent_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        clip_image(image_path, os.path.join(output_dir, os.path.basename(image_path)), extent)
    
    return True
        
if __name__ == "__main__":
    # 需裁减的地理范围，像元大小与裁剪大小
    area = [116.0, 117.5, 39.5, 41.0]
    pixel_size = [0.000001, 0.000001]
    size = [2048, 2048]
    
    # 计算两个方向的列表
    num_x = int((area[1] - area[0]) / pixel_size[0] / size[0])
    num_y = int((area[3] - area[2]) / pixel_size[1] / size[1])
    x_list = area[0] + pixel_size[0] * size[0] * (np.arange(0, num_x, 1))
    y_list = area[2] + pixel_size[1] * size[1] * (np.arange(0, num_y, 1))
    
    # 裁剪影像
    time1_dir = ""
    time2_dir = ""
    all_clipped_num = 0
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            extent = [x, x + pixel_size[0] * size[0], y, y + pixel_size[1] * size[1]]
            if clip_change_image(time1_dir, time2_dir, extent):
                all_clipped_num += 1
            print("finished: ", i * num_y + j + 1, "/", num_x * num_y)
    print("all_clipped_num: ", all_clipped_num)