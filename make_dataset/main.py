import glob
from shp_functions import merge_shp, shp2raster, trans_shp
from raster_functions import searchShpByRaster, clipMaskByImg
from configs import get_colormap
if __name__ == '__main__':
    print("begin:")
    shp_dir = "shapefile/" # the path of all shp files
    file = 'I:/WHU_WY/image_finish/20DEC01030057-M2AS-013317249080_04_P002_FUS_DOM.tif' # the path of the image, which is used to create the dataset
    
    # 查询覆盖的shp
    print("1.begin to search")
    shp_list, pixel_size = searchShpByRaster(file, shp_dir)
    if len(shp_list) == 0:
        print("mistake appear: no shp file is found")
        exit(0)
    print("finish search")
    
    # 合并覆盖shp
    print("2.begin merge shp")
    file_name = file.split("\\")[-1].split(".")[0]
    shp_file = merge_shp(shp_list, file_name)
    print("finish merge shp")
    
    # 类别转换
    print("3.begin translate shp")
    trans_shp(shp_file)
    print("finish translate shp")
    
    # shp转栅格
    print("4.begin shp2raster")
    output_raster = shp_file.split(".")[0] + '.tif'
    colormap = get_colormap()
    shp2raster(shp_file, output_raster, pixel_size, colormap)
    print("finish shp2raster")
    
    # 裁剪
    print("5.begin clip")
    clipMaskByImg(file, output_raster)
    print("finish clip")
    
    print("all finished")