import glob
import rasterio
import cv2
from shp_functions import merge_shp, shp2raster, trans_shp
from raster_functions import *
from configs import get_colormap, get_guiyang_labelmap
from clip_change import *
from utils import *
from clip_classify import main as clip_classify

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

    ori_img_path = "/media/dell/DATA/wy/data/guiyang/合并影像/西秀/2021_nir/"
    ori_sar_path = "/media/dell/DATA/wy/data/guiyang//sar/" #十二个月份
    ori_label_path = "/media/dell/DATA/wy/data/guiyang/标签/分类/西秀/mask_crop/"
    ori_label3_path = "/media/dell/DATA/wy/data/guiyang/标签/分类/西秀/mask_agri/"
    ori_builtup_label_path = "/media/dell/DATA/wy/data/guiyang/标签/变化检测/剑河/2021标签/label_2021_cons_trans.tif"
    save_path = "/media/dell/DATA/wy/data/guiyang/数据集/v2/分类/"
    
    ori_img_list = glob.glob(ori_img_path + "*.tif")
    flag = 0
    for ori_img in ori_img_list:
        # ori_img = "/media/dell/DATA/wy/data/guiyang/合并影像/剑河/2021_nir/GF71.tif"
        print("begin:", ori_img)
        new_dir = save_path + ori_img.split("/")[-1].split(".")[0]
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        # 裁剪施工区域标签
        builtup_clip_save_path = os.path.join(new_dir, "builtup_clip")
        if not os.path.exists(builtup_clip_save_path):
            os.mkdir(builtup_clip_save_path)
        builtup_label_path = clip_builtup(ori_img, ori_builtup_label_path, builtup_clip_save_path)
        
        # 裁剪标签
        label_clip_save_path = os.path.join(new_dir, "label_clip")
        if not os.path.exists(label_clip_save_path):
            os.mkdir(label_clip_save_path)

        ori_label = os.path.join(ori_label_path, os.path.basename(ori_img))
        label_path = clip_label(ori_img, ori_label, label_clip_save_path)
        
        # 附加施工区域标签
        builtup_label_save_path = os.path.join(new_dir, "builtup_add")
        if not os.path.exists(builtup_label_save_path):
            os.mkdir(builtup_label_save_path)
        label_path = add_builtup_label(label_path, builtup_label_path, builtup_label_save_path)
        
        # 裁剪非农化标签
        label3_clip_save_path = os.path.join(new_dir, "label3_clip")
        if not os.path.exists(label3_clip_save_path):
            os.mkdir(label3_clip_save_path)

        ori_label = os.path.join(ori_label3_path, os.path.basename(ori_img))
            
        label3_path = clip_label(ori_img, ori_label, label3_clip_save_path)
        
        
        # 附加非农化的施工区域标签标签
        builtup_label_save_path = os.path.join(new_dir, "builtup3_add")
        if not os.path.exists(builtup_label_save_path):
            os.mkdir(builtup_label_save_path)
        label_path = add_builtup_label(label3_path, builtup_label_path, builtup_label_save_path)
        
        # 裁剪sar
        # sar_clip_save_path = os.path.join(new_dir, "sar_clip")
        # if not os.path.exists(sar_clip_save_path):
        #     os.mkdir(sar_clip_save_path)
        # sar_path_list = []
        # for dir, _, files in os.walk(ori_sar_path):
        #     for file in files:
        #         if file.split(".")[0] == os.path.basename(ori_img).split(".")[0]:
        #             ori_sar = os.path.join(dir, file)
        #             sar_path = clip_sar(ori_img, ori_sar, sar_clip_save_path)
        #             sar_path_list.append(sar_path)
                    
        # 分块
        # split_img_label(ori_img, label_path, label3_path, sar_name_list=sar_path_list, split_size=1024, overlap_size=0, save_path=new_dir)

    
    # 根据需要转换标签
    # print("1.begin to trans tif label")
    # label_path = add_builtup_label(ori_label_path, builtup_label_path)
    # label_path = trans_raster_label(ori_label_path, get_guiyang_labelmap())
    # print("trans finished")
    
    # 检查栅格和标签的地理参数是否一致，返回相应的栅格和标签路径(假设sar和img的地理参数一致)
    # print("2.begin to check geo params")
    # img_list, label_list = check_geo_params(ori_img_path, label_path)
    # print("check finished")
    
    # 裁剪
   
# 贵阳变化检测数据条件
def guiyang_change():
    # 前置操作：包括查询覆盖的shp、合并覆盖shp、类别转换、shp转栅格
    # 当前数据条件：得到了栅格和对应标签，但两时像栅格范围没有对齐
    
    # TODO: 可能需要考虑nodata的情况
    # 需裁减的地理范围，像元大小与裁剪大小
    t1_label_path = ""
    t2_label_path = ""
    t1_dataset = gdal.Open(t1_label_path)
    t2_dataset = gdal.Open(t2_label_path)
    t1_geo = t1_dataset.GetGeoTransform()
    t2_geo = t2_dataset.GetGeoTransform()
    
    left_up_x = max(t1_geo[0], t2_geo[0])
    left_down_x = min(t1_geo[0] + t1_geo[1] * t1_dataset.RasterXSize, t2_geo[0] + t2_geo[1] * t2_dataset.RasterXSize)
    right_up_y = min(t1_geo[3], t2_geo[3])
    right_down_y = max(t1_geo[3] + t1_geo[5] * t1_dataset.RasterYSize, t2_geo[3] + t2_geo[5] * t2_dataset.RasterYSize)
    
    area = [left_up_x, left_down_x, right_up_y, right_down_y]
    pixel_size = [t1_geo[1], t1_geo[5]]
    size = [512, 512]
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

def make_sar_image(sar_path, img_path):
    for dir_name, _, file_list in os.walk(sar_path):
        for file in file_list:
            if file.endswith("GF71.tif"):
                file_path = os.path.join(dir_name, file)
                # resample
                resample_path = file_path.replace(".tif", "_resample.tif")
                resample_command = "gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4524 -tr 0.65 0.65 -r near -of GTiff {} {}".format(file_path, resample_path)
                print(os.popen(resample_command).read())
                
                # clip
                img_dataset = gdal.Open(img_path)
                img_geo = img_dataset.GetGeoTransform()
                img_extent = [img_geo[0], img_geo[0] + img_geo[1] * img_dataset.RasterXSize, img_geo[3] + img_geo[5] * img_dataset.RasterYSize, img_geo[3]]
                clip_path = file_path.replace(".tif", "_clip.tif")
                clip_command = "gdal_translate -projwin {} {} {} {} -of GTiff {} {}".format(img_extent[0], img_extent[3], img_extent[1], img_extent[2], resample_path, clip_path)
                print(os.popen(clip_command).read())
                print("finished: ", file_path)
                
def guiyang_change_Li():
    time1_label_dir = "/media/dell/DATA/wy/data/guiyang/变化检测/西秀区/2020标签裁剪/施工/"
    time1_img_dir = "/media/dell/DATA/wy/data/guiyang/变化检测/西秀区/2020裁剪/"
    time2_label_dir = "/media/dell/DATA/wy/data/guiyang/变化检测/西秀区/2021标签裁剪/施工/"
    time2_img_dir = "/media/dell/DATA/wy/data/guiyang/变化检测/西秀区/2021裁剪/"
    
    time1_label_list = glob.glob(os.path.join(time1_label_dir, "*.tif"))
    time1_img_list = glob.glob(os.path.join(time1_img_dir, "*.tif"))
    time2_label_list = glob.glob(os.path.join(time2_label_dir, "*.tif"))
    time2_img_list = glob.glob(os.path.join(time2_img_dir, "*.tif"))
    for i, file_path in enumerate(time1_label_list):
        file_name = os.path.basename(file_path)
        id = file_name.split("_")[1]
        # 获取路径
        time1_label_path = get_id_names(id, time1_label_list)
        time1_img_path = get_id_names(id, time1_img_list)
        time2_label_path = get_id_names(id, time2_label_list)
        time2_img_path = get_id_names(id, time2_img_list)
        
        if len(time1_label_path) == 0 or len(time1_img_path) == 0 or len(time2_label_path) == 0 or len(time2_img_path) == 0:
            continue
        # 过滤
        time1_label = gdal.Open(time1_label_path[0]).ReadAsArray()
        time2_label = gdal.Open(time2_label_path[0]).ReadAsArray()
        time1_img = gdal.Open(time1_img_path[0]).ReadAsArray()
        time2_img = gdal.Open(time2_img_path[0]).ReadAsArray()
        time1_img = time1_img.transpose(1, 2, 0)
        time2_img = time2_img.transpose(1, 2, 0)
        a = np.where(np.all(time1_img == [0, 0, 0, 0], axis=-1))[:2]
        b = np.where(np.all(time2_img == [0, 0, 0, 0], axis=-1))[:2]
        if len(a[0])>50 or len(b[0])>50:
            continue
        
        # tif2png
        # time1_label_png_path = label_tif2png(time1_label_path[0])
        # time1_img_png_path = img_save(time1_img_path[0])
        time2_label_png_path = label_tif2png(time2_label_path[0])
        # time2_img_png_path = img_save(time2_img_path[0])
        
        print("finished: {}/{}".format(i, len(time1_label_list)))
    
    return 0
def cumulative_count_cut(path, file_type):
    image_list = glob.glob(os.path.join(path, "*.{}".format(file_type)))
    for i, file_path in enumerate(image_list):
        print("begin {}/{}".format(i, len(image_list)))
        img_dataset = gdal.Open(file_path)

        img_array = img_dataset.ReadAsArray()
        R = img_array[0].astype(np.float32)
        G = img_array[1].astype(np.float32)
        B = img_array[2].astype(np.float32)
        NIR = img_array[3].astype(np.float32)
        index = np.where(np.all(img_array.transpose(1, 2, 0) == [0, 0, 0, 0], axis=-1))[:2]
        R[index] = np.nan
        G[index] = np.nan
        B[index] = np.nan
        NIR[index] = np.nan
        R_max = np.nanpercentile(R, 75)
        R_min = np.nanpercentile(R, 25)
        R[R > R_max] = R_max
        R[R < R_min] = R_min
        R = (R - R_min) / (R_max - R_min) * 255
        
        G_max = np.nanpercentile(G, 75)
        G_min = np.nanpercentile(G, 25)
        G[G > G_max] = G_max
        G[G < G_min] = G_min
        G = (G - G_min) / (G_max - G_min) * 255
        
        B_max = np.nanpercentile(B, 75)
        B_min = np.nanpercentile(B, 25)
        B[B > B_max] = B_max
        B[B < B_min] = B_min
        B = (B - B_min) / (B_max - B_min) * 255
        
        NIR_max = np.nanpercentile(NIR, 75)
        NIR_min = np.nanpercentile(NIR, 25)
        NIR[NIR > NIR_max] = NIR_max
        NIR[NIR < NIR_min] = NIR_min
        NIR = (NIR - NIR_min) / (NIR_max - NIR_min) * 255
        
        save_path = file_path.replace(".{}".format(file_type), "_cut.tif")
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(save_path, img_dataset.RasterXSize, img_dataset.RasterYSize, 4, gdal.GDT_Byte)
        outdata.SetGeoTransform(img_dataset.GetGeoTransform())
        outdata.SetProjection(img_dataset.GetProjection())
        outdata.GetRasterBand(1).WriteArray(R.astype(np.uint8))
        outdata.GetRasterBand(2).WriteArray(G.astype(np.uint8))
        outdata.GetRasterBand(3).WriteArray(B.astype(np.uint8))
        outdata.GetRasterBand(4).WriteArray(NIR.astype(np.uint8))
        outdata.FlushCache()
        outdata = None

def set_nodata_value2nan(path):
    image_list = glob.glob(os.path.join(path, "*.tif"))
    for i, file_path in enumerate(image_list):
        print("begin {}/{}".format(i, len(image_list)))
        img_dataset = gdal.Open(file_path)
        # print(img_dataset.GetRasterBand(1).GetNoDataValue())
        # img_dataset.GetRasterBand(1).SetNoDataValue()
def guiyang_img_trans(path):
    image_list = glob.glob(os.path.join(path, "*.png"))
    for i, file_path in enumerate(image_list):
        png2tif(file_path)
        
def add_nir_channel(rgb_path, nir_path, save_dir):
    rgb_path = os.path.join(rgb_path, "*.tif")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in glob.glob(rgb_path):
        print("begin: {}".format(file))
        nir_file = os.path.join(nir_path, os.path.basename(file))
        img_dataset = gdal.Open(file)
        img_array = img_dataset.ReadAsArray()
        nir_dataset = gdal.Open(nir_file)
        nir_array = nir_dataset.ReadAsArray()
        NIR = nir_array[3].astype(np.float32)
        index = np.where(np.all(nir_array.transpose(1, 2, 0) == [0, 0, 0, 0], axis=-1))[:2]
        NIR[index] = np.nan
        NIR[index] = np.nan
        NIR_max = np.nanpercentile(NIR, 98)
        NIR_min = np.nanpercentile(NIR, 2)
        NIR[NIR > NIR_max] = NIR_max
        NIR[NIR < NIR_min] = NIR_min
        NIR = (NIR - NIR_min) / (NIR_max - NIR_min) * 255
        
        NIR = NIR.astype(np.uint8)
        if img_array.shape[-2:] != NIR.shape:
            NIR = cv2.resize(NIR, (img_array.shape[-1], img_array.shape[-2]))
        img_array = np.concatenate((img_array, NIR[np.newaxis, :, :]), axis=0)
        save_path = os.path.join(save_dir, os.path.basename(file))
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(save_path, img_dataset.RasterXSize, img_dataset.RasterYSize, 4, gdal.GDT_Byte)
        outdata.SetGeoTransform(img_dataset.GetGeoTransform())
        outdata.SetProjection(img_dataset.GetProjection())
        outdata.GetRasterBand(1).WriteArray(img_array[0])
        outdata.GetRasterBand(2).WriteArray(img_array[1])
        outdata.GetRasterBand(3).WriteArray(img_array[2])
        outdata.GetRasterBand(4).WriteArray(img_array[3])
        outdata.FlushCache()
        outdata = None
        
def clip_xixiu(img_dir, rgb_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img_list = glob.glob(os.path.join(img_dir, "*.tif"))
    for i, img_path in enumerate(img_list):
        print("begin {}/{}".format(i, len(img_list)))
        img_name = os.path.basename(img_path)
        rgb_path = os.path.join(rgb_dir, img_name)
        if not os.path.exists(rgb_path):
            print("error: {}".format(rgb_path))
            continue
        img_dataset = gdal.Open(img_path)
        img_geo = img_dataset.GetGeoTransform()
        img_extent = [img_geo[0], img_geo[0] + img_geo[1] * img_dataset.RasterXSize, img_geo[3] + img_geo[5] * img_dataset.RasterYSize, img_geo[3]]
        clip_command = "gdal_translate -projwin {} {} {} {} -of GTiff {} {}".format(img_extent[0], img_extent[3], img_extent[1], img_extent[2], rgb_path, os.path.join(save_dir, img_name))
        os.system(clip_command)
        
def trans_crs(img_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img_list = glob.glob(os.path.join(img_dir, "*.tif"))
    for i, img_path in enumerate(img_list):
        print("begin {}/{}".format(i, len(img_list)))
        result_path = os.path.join(save_dir, os.path.basename(img_path))
        trans_command = 'gdal_translate -a_srs EPSG:4524 -of GTiff {} {}'.format(img_path, result_path)
        os.system(trans_command)
        
def change_sar_srs(sar_dir):
    for dir, _, files in os.walk(sar_dir):
        for file in files:
            if file.endswith(".tiff"):
                save_dir = os.path.join(dir, "warp")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                gdal_swarp_to_4524(os.path.join(dir, file), os.path.join(save_dir, file))
                
def change_xixiu_srs(data_dir):
    for dir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".tif"):
                save_dir = os.path.join(dir, "warp")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                gdal_swarp_to_4524(os.path.join(dir, file), os.path.join(save_dir, file))
if __name__ == '__main__':
    # 改变影像投影坐标
    # change_sar_srs("/media/dell/DATA/wy/data/guiyang/sar/sar/")
    # change_xixiu_srs("/media/dell/DATA/wy/data/guiyang/原始影像/西秀/2020/")
    # guangdong()
    # guiyang()
    # guiyang_change_Li()
    # guiyang_change()
    # gdal_translate("/media/dell/DATA/wy/data/guiyang/剑河/光学影像2021/")
    # make_sar_image("/media/dell/DATA/wy/data/guiyang/剑河/sar/", "/media/dell/DATA/wy/data/guiyang/剑河/光学影像2021/GF71.png")
    # make_label_dataset("/media/dell/DATA/wy/data/guiyang/剑河/mask_crop/GF71.png", "/media/dell/DATA/wy/data/guiyang/剑河/光学影像2021/GF71.png")
    
    # img_path = "/media/dell/DATA/wy/data/guiyang/剑河/光学影像2021/"
    # img_list = glob.glob(os.path.join(img_path, "*.png"))
    # for img in img_list:
    #     change_img_scale(img)
    
    # 设置nodata值
    # set_nodata_value2nan("/media/dell/DATA/wy/data/guiyang/西秀/2022年重采样/")
    
    # 色彩拉伸
    # cumulative_count_cut("/media/dell/DATA/wy/data/guiyang/剑河/2020影像/", file_type='tif')
    
    # png2tif
    # guiyang_img_trans("/media/dell/DATA/wy/data/guiyang/剑河/光学影像2021/")
    
    # 裁剪西秀影像
    # clip_xixiu("/media/dell/DATA/wy/data/guiyang/原始影像/西秀/2021/warp/", "/media/dell/DATA/wy/data/guiyang/RGB影像/西秀/2021/", "/media/dell/DATA/wy/data/guiyang/RGB影像/西秀/2020_clip/")
    
    # 通道叠加
    add_nir_channel("/media/dell/DATA/wy/data/guiyang/RGB影像/西秀/2021_clip/", "/media/dell/DATA/wy/data/guiyang/原始影像/西秀/2021/warp/", save_dir="/media/dell/DATA/wy/data/guiyang/RGB影像/西秀/2021_nir/")
    
    # 裁剪分类数据
    # clip_classify