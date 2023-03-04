from functions import *

if __name__ == "__main__":
    # extract_feature("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县.shp", "/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp", "NAME", "西秀区")
    # warp_sar("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp", "/media/dell/DATA/wy/data/guiyang/sar/西秀/2021/")
    # warp_sar("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp", "/media/dell/DATA/wy/data/guiyang/sar/西秀/2022/")
    # warp_sar("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_剑河.shp", "/media/dell/DATA/wy/data/guiyang/sar/剑河/2021/")
    # warp_sar("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_剑河.shp", "/media/dell/DATA/wy/data/guiyang/sar/剑河/2022/")
    clip_sar("/media/dell/DATA/wy/data/guiyang/sar/西秀/2021/", "/media/dell/DATA/wy/data/guiyang/合并影像/西秀/2021_split/", "/media/dell/DATA/wy/data/guiyang/sar/西秀/2021_split/")
    # fill_nodata_erea("/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp", "/media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀_fill.shp")
    # split_big_tif("/media/dell/DATA/wy/data/guiyang/合并影像/剑河/2022_nir/merge_cut.tif", "/media/dell/DATA/wy/data/guiyang/合并影像/剑河/2022_split/", 1024)
    # copy_files_from_another("/media/dell/DATA/wy/data/guiyang/合并影像/西秀/2022_split/", "/media/dell/DATA/wy/data/guiyang/合并影像/西秀/2021_split/")
    
# 切割图像"gdalwarp -cutline /media/dell/DATA/wy/data/guiyang/贵州省/贵州省_县_西秀.shp -crop_to_cutline merge.tif merge_cut.tif"