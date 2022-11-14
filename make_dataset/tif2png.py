import glob
from osgeo import gdal

def tif2png(tif_name):
    options = gdal.TranslateOptions(format='PNG', bandList=[])
    gdal.Translate(tif_name.replace('.tif', '.png'), tif_name, options=options)
def tif2bmp(tif_name):
    options = gdal.TranslateOptions(format='BMP', bandList=[])
    gdal.Translate(tif_name.replace('.tif', '.bmp'), tif_name, options=options)

if __name__ == "__main__":
    tif_dir = "I:/WHU_WY/label/"
    file_list = glob.glob(('{}*.tif'.format(tif_dir)))
    for file_name in file_list:
        tif2png(file_name)
