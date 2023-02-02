import glob
from shp_functions import merge_shp, shp2raster, trans_shp
from raster_functions import *
from configs import get_colormap, get_guiyang_labelmap
from clip_change import *

# 广州数据条件
def guangdong():
    print("begin:")
    shp_dir = "shapefile/" # the path of all shp files
    file = 'I:/WHU_WY/image_finish/20DEC01030057-M2AS-013317249080_04_P002_FUS_DOM.tif' # the path of the image, which is used to create the dataset
    
    # 查询覆盖的shp
    print("1.begin to search")
    shp_list, pixel_size = search_shp_by_raster(file, shp_dir)
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
    clip_mask_by_img(file, output_raster)
    print("finish clip")
    
    print("all finished")
    
# 贵阳非农化非粮化数据条件
def guiyang():
    # 前置操作：包括查询覆盖的shp、合并覆盖shp、类别转换、shp转栅格
    # 当前数据条件：得到了栅格和对应标签，但需要对标签进行转换
    ori_img_path = ""
    ori_sar_path = "" #十二个月份
    ori_label_path = ""
    
    # 根据需要转换标签
    print("1.begin to trans tif label")
    label_path = trans_raster_label(ori_label_path, get_guiyang_labelmap())
    print("trans finished")
    
    # 检查栅格和标签的地理参数是否一致，返回相应的栅格和标签路径(假设sar和img的地理参数一致)
    print("2.begin to check geo params")
    img_list, label_list = check_geo_params(ori_img_path, label_path)
    print("check finished")
    
    # 裁剪
    print("3.begin clip")
    split_img_label(img_list, label_list, split_size=512, overlap_size=0, sar_path=ori_sar_path, save_path="")

# 贵阳变化检测数据条件
def guiyang_change():
    # 前置操作：包括查询覆盖的shp、合并覆盖shp、类别转换、shp转栅格
    # 当前数据条件：得到了栅格和对应标签，但两时像栅格范围没有对齐
    
    # TODO: 可能需要考虑nodata的情况
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
    
if __name__ == '__main__':
    # guangdong()
    # guiyang()