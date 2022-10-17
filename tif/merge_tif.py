import glob
import os

def merge_tif(tif_dir):
    files_to_merge = glob.glob(tif_dir + '*.tif')
    files_string = " ".join(files_to_merge)
    print(files_string)
    command = "python gdal_merge.py -o {}merge.tif -of gtiff ".format(tif_dir) + files_string
    print(os.popen(command).read())