import os
from osgeo import gdal
import numpy as np
import time
def remap_values_in_folder(input_dir, output_dir, value_map):
    def remap_values(input_file, output_file, value_map):
        # 打开输入TIF文件
        ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError(f"Failed to open input file: {input_file}")

        # 读取数据
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()

        # 使用value_map进行转换
        for k, v in value_map.items():
            data[data == k] = v

        # 保存结果到新的文件
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_file, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(data)

        # 释放资源
        ds = None
        out_ds = None

    # 创建输出文件夹，如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入文件夹中的所有.tif文件
    begin = time.time()
    image_num = len(os.listdir(input_dir))
    for i, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.tif'):
            now = time.time()
            avg_time = (now - begin) / (i + 1)
            rest_time = avg_time * (image_num - i - 1)
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            remap_values(input_file, output_file, value_map)
            print(f"Remapped {input_file} to {output_file}, rest time: {rest_time:.2f}s")

# 使用示例
input_directory = "/media/dell/DATA/wy/data/guiyang/guizhou_dataset/val/lc/"
output_directory = "/media/dell/DATA/wy/data/guiyang/guizhou_dataset/val/vege_lc/"
value_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2}
remap_values_in_folder(input_directory, output_directory, value_map)