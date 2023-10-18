import os
from osgeo import gdal

def convert_folder_to_gray(input_dir, output_dir):
    def band_to_gray(input_file, output_file):
        ds = gdal.Open(input_file, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError(f"Failed to open input file: {input_file}")

        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        bands = ds.RasterCount

        if bands < 4:
            raise ValueError("Input image should have at least 4 bands.")

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_file, xsize, ysize, 1, gdal.GDT_Byte)
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())

        gray_band = out_ds.GetRasterBand(1)
        band_data = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(4)]
        gray_data = (band_data[0] + band_data[1] + band_data[2] + band_data[3]) / 4
        gray_band.WriteArray(gray_data)

        out_ds.FlushCache()
        ds = None
        out_ds = None

    # 创建输出文件夹，如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    import time
    origin_time = time.time()
    # 遍历输入文件夹中的所有.tif文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            now_time = time.time()
            avg_time = (now_time-origin_time)/(os.listdir(input_dir).index(filename)+1)
            remain_time = avg_time*(len(os.listdir(input_dir))-os.listdir(input_dir).index(filename))
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            band_to_gray(input_file, output_file)
            print(f"Processed {filename}, remain time: {remain_time}")

# 使用示例
input_directory = "/media/dell/DATA/wy/data/guiyang/guizhou_dataset/test/opt"
output_directory = "/media/dell/DATA/wy/data/guiyang/guizhou_dataset/test/gray"
convert_folder_to_gray(input_directory, output_directory)