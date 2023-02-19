from osgeo import ogr
from osgeo import gdal
import os
import numpy as np
import glob
def extract_feature(input_shp, output_shp, field_name, field_value):
    # 打开原始shp文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    input_ds = driver.Open(input_shp, 0)
    if input_ds is None:
        print('Could not open input shapefile')
        return

    # 获取原始shp文件的第一个图层
    layer = input_ds.GetLayer(0)

    # 根据筛选条件构建查询语句
    query = f"{field_name} = '{field_value}'"

    # 进行要素筛选
    layer.SetAttributeFilter(query)

    # 创建新的shapefile文件
    output_ds = driver.CreateDataSource(output_shp)
    if output_ds is None:
        print('Could not create output shapefile')
        return

    # 从原始图层中复制筛选出的要素到新图层中
    output_layer = output_ds.CopyLayer(layer, 'output_layer')

    # 关闭文件句柄
    input_ds = None
    output_ds = None

    print(f"Successfully extracted features to {output_shp}")
    
def point_in_shapefile(shp, point):
    # 获取第一个图层
    layer = shp.GetLayer(0)

    # 创建要素的几何
    point_geom = ogr.Geometry(ogr.wkbPoint)
    point_geom.AddPoint(point[0], point[1])

    # 遍历所有要素，检查点是否在其中
    for feature in layer:
        if point_geom.Within(feature.GetGeometryRef()):
            return True

    # 如果点不在任何要素中，返回 False
    return False

def nodata_to_polygon(tif_file, output_shp):
    # 打开 tif 文件
    tif_ds = gdal.Open(tif_file)
    if tif_ds is None:
        print('无法打开文件 {}'.format(tif_file))
        return

    # 获取 nodata 值
    nodata = tif_ds.GetRasterBand(1).GetNoDataValue()

    # 创建临时文件来存储 rasterized image
    temp_raster = 'temp.tif'
    temp_shp = 'temp.shp'

    # 将 nodata 值的区域设为 -9999
    gdal.Translate(temp_raster, tif_file, options=f'-a_nodata -9999')

    # 使用 Rasterize 函数将 -9999 值转换为矢量
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    temp_shp_ds = shp_driver.CreateDataSource(temp_shp)
    temp_shp_lyr = temp_shp_ds.CreateLayer('polygonized', srs=tif_ds.GetProjectionRef())
    raster_band = tif_ds.GetRasterBand(1)
    gdal.Polygonize(raster_band, None, temp_shp_lyr, 0)

    # 获取 polygonized shapefile 中的要素
    feature = temp_shp_lyr.GetFeature(0)
    geom = feature.GetGeometryRef()

    # 创建输出 shapefile
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_ds = output_driver.CreateDataSource(output_shp)
    output_lyr = output_ds.CreateLayer('nodata_polygon', srs=tif_ds.GetProjectionRef())
    field_defn = ogr.FieldDefn('id', ogr.OFTInteger)
    output_lyr.CreateField(field_defn)

    # 将 polygon 添加到输出图层中
    output_feat = ogr.Feature(output_lyr.GetLayerDefn())
    output_feat.SetGeometry(geom)
    output_feat.SetField('id', 1)
    output_lyr.CreateFeature(output_feat)

    # 关闭文件和清理临时文件
    del tif_ds
    del temp_shp_ds
    del output_ds
    gdal.Unlink(temp_raster)
    gdal.Unlink(temp_shp)

def intersect_vectors(infile1, infile2, outfile):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    in_ds1 = ogr.Open(infile1)
    in_layer1 = in_ds1.GetLayer()
    in_ds2 = ogr.Open(infile2)
    in_layer2 = in_ds2.GetLayer()
    if in_layer1.GetSpatialRef() != in_layer2.GetSpatialRef():
        print("Error: Spatial reference systems do not match.")
        return
    spatial_ref = in_layer1.GetSpatialRef()
    if driver.Open(outfile):
        driver.DeleteDataSource(outfile)
    out_ds = driver.CreateDataSource(outfile)
    out_layer = out_ds.CreateLayer('intersection', spatial_ref, ogr.wkbPolygon)
    out_layer_defn = out_layer.GetLayerDefn()
    in_layer1.SetSpatialFilter(in_layer2.GetExtent())
    for in_feat1 in in_layer1:
        in_geom1 = in_feat1.GetGeometryRef()
        in_layer2.SetSpatialFilter(in_geom1)
        for in_feat2 in in_layer2:
            in_geom2 = in_feat2.GetGeometryRef()
            if in_geom1.Intersects(in_geom2):
                out_geom = in_geom1.Intersection(in_geom2)
                out_feat = ogr.Feature(out_layer_defn)
                out_feat.SetGeometry(out_geom)
                out_layer.CreateFeature(out_feat)
                out_feat = None
        in_layer2.ResetReading()
    in_layer1.ResetReading()
    in_ds1 = None
    in_ds2 = None
    out_ds = None

def warp_sar(shp_file_path, sar_dir):
    sar_list = glob.glob(os.path.join(sar_dir, '*.tif'))
    for sar_file_path in sar_list:
        out_put_file = sar_file_path.replace('.tif', '_cut.tif')
        warp_command = f'gdalwarp -cutline {shp_file_path} -crop_to_cutline {sar_file_path} {out_put_file}'
        os.system(warp_command)

def fill_nodata_erea(ori_tif_path, img_dir, shp_path, save_tif_path):
    ori_img_dataset = gdal.Open(ori_tif_path)
    ori_array = ori_img_dataset.ReadAsArray()
    ori_geo = ori_img_dataset.GetGeoTransform()
    index = np.where(np.all(ori_array.transpose(1, 2, 0) == [0, 0, 0, 0], axis=-1))[:2]
    # 打开 shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(shp_path, 0)
    for i, j in zip(index[0], index[1]):
        point_x = ori_geo[0] + ori_geo[1] * j
        point_y = ori_geo[3] + ori_geo[5] * i
        point = [point_x, point_y]
        if point_in_shapefile(shp, point):
            img_name = os.path.join(img_dir, f"{i}_{j}.tif")
            img_dataset = gdal.Open(img_name)
            img_array = img_dataset.ReadAsArray()
            ori_array[:, i, j] = img_array[:, 0, 0]
