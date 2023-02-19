from functions import *

if __name__ == "__main__":
    # extract_feature("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县.shp", "/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp", "NAME", "西秀区")
    # warp_sar("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp", "/media/dell/DATA/wy/data/guiyang/sar/西秀/2021/")
    # warp_sar("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp", "/media/dell/DATA/wy/data/guiyang/sar/西秀/2022/")
    # warp_sar("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_剑河.shp", "/media/dell/DATA/wy/data/guiyang/sar/剑河/2021/")
    # warp_sar("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_剑河.shp", "/media/dell/DATA/wy/data/guiyang/sar/剑河/2022/")
    # fill_nodata_erea("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp", "/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀_fill.shp")
    
# 切割图像"gdalwarp -cutline /media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp -crop_to_cutline merge.tif merge_cut.tif"