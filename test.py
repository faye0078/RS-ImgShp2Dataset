import numpy as np
import os
from osgeo import gdal
def is_exist_zero_pixel_1(array, max_num):
    num = sum(sum(array == 0))
    num_15 = sum(sum(array == 15))
    num_128 = sum(sum(array == 128))
    if num > max_num or num_15 > max_num or num_128 > max_num:
        return True
    else:
        return False

def fliter_data(img_path, label1_path, label2_path, label3_path, sar_path):
    for dir, _, file_list in os.walk(label1_path):
        for file in file_list:
            if file.endswith(".tif"):
                label1_full_path = os.path.join(dir, file)
                image_full_path = os.path.join(label1_path, file).replace('label1', 'image')
                label2_full_path = os.path.join(label2_path, file).replace('label1', 'label2')
                label3_full_path = os.path.join(label3_path, file).replace('label1', 'label3')
                sar_full_path = []
                sar_base = dir.replace('label1', 'sar')
                for i in range(12):
                    each_sar_path = os.path.join(sar_base, str(i+1), file).replace('label1', 'sar')
                    sar_full_path.append(each_sar_path)
                
                if is_exist_zero_pixel_1(np.array(gdal.Open(label1_full_path).ReadAsArray()), 100):
                    os.remove(image_full_path)
                    os.remove(label1_full_path)
                    os.remove(label2_full_path)
                    os.remove(label3_full_path)
                    for each_sar in sar_full_path:
                        os.remove(each_sar)      
            print("finish {}/{}".format(file_list.index(file), len(file_list)))
                        
fliter_data("/media/dell/DATA/wy/data/guiyang/数据集/v2/分类/西秀/image/", 
            "/media/dell/DATA/wy/data/guiyang/数据集/v2/分类/西秀/label1/", 
            "/media/dell/DATA/wy/data/guiyang/数据集/v2/分类/西秀/label2/",
            "/media/dell/DATA/wy/data/guiyang/数据集/v2/分类/西秀/label3/",
            "/media/dell/DATA/wy/data/guiyang/数据集/v2/分类/西秀/sar/")