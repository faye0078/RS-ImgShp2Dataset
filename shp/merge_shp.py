import glob
import os
def merge_shp(shp_dir):
    files_to_merge = glob.glob(shp_dir + '*.shp')
    files_string = " ".join(files_to_merge)
    print(files_string)
    command = "C:/Users/505/Anaconda3/envs/point_test/python.exe ogrmerge.py -single -o {}merged.shp ".format(shp_dir) + files_string
    print(os.popen(command).read())
    # print(os.popen('ls {}'.format(shp_dir)).read())

if __name__ == "__main__":
    # root = ""
    # dir_list = os.listdir(root)
    # for dir in dir_list:
    #     merge_shp(dir)
    dir = "J:/GuangdongSHP/splitSHP/2/"
    merge_shp(dir)