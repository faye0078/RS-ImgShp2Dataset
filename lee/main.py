from osgeo import gdal,osr
import geopandas as gpd
from shapely.geometry import Point
import random
import pickle
from matplotlib import pyplot
import numpy
import os
from PIL import Image

def getSRSPair(dataset):
    prosrs=osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs=prosrs.CloneGeogCS()
    return prosrs,geosrs

def lonlat2geo(dataset,lon,lat):
    prosrs,geosrs=getSRSPair(dataset)
    ct=osr.CoordinateTransformation(geosrs,prosrs)
    coords=ct.TransformPoint(lon,lat)
    return coords[:2]

def geo2lonlat(dataset,x,y):
    prosrs,geosrs=getSRSPair(dataset)
    ct=osr.CoordinateTransformation(prosrs,geosrs)
    coords=ct.TransformPoint(x,y)
    return coords[:2]

def generate_points(path_shp,size,pixel_size,ref):


    path_image=ref


    dataset=gdal.Open(path_image)


    shp=gpd.read_file(path_shp)

    for i in range(len(shp)):
        id=shp.iloc[i]['XZQDM']
        if id=='520402':
            polygon_xixiu=shp.iloc[i]['geometry']
        if id=='522629':
            polygon_jianhe=shp.iloc[i]['geometry']

    x1,y1=lonlat2geo(dataset,polygon_xixiu.bounds[0],polygon_xixiu.bounds[1])
    x2,y2=lonlat2geo(dataset,polygon_xixiu.bounds[2],polygon_xixiu.bounds[3])
    bound_xixiu=[x1,x2,y1,y2]

    x1,y1=lonlat2geo(dataset,polygon_jianhe.bounds[0],polygon_jianhe.bounds[1])
    x2,y2=lonlat2geo(dataset,polygon_jianhe.bounds[2],polygon_jianhe.bounds[3])
    bound_jianhe=[x1,x2,y1,y2]





    result={'xixiu':None,'jianhe':None}

    #xixiu
    points=[]
    for i in numpy.arange(bound_xixiu[0],bound_xixiu[1],size*pixel_size):
        for j in numpy.arange(bound_xixiu[2],bound_xixiu[3],size*pixel_size):
            x,y=round(i,5),round(j,5)
            lon,lat=geo2lonlat(dataset,x,y)
            point_lonlat=Point(lon,lat)
            if polygon_xixiu.contains(point_lonlat):
                points.append([lon,lat])
    result['xixiu']=points
    print(len(points))


    points=[]
    for i in numpy.arange(bound_jianhe[0],bound_jianhe[1],size*pixel_size):
        for j in numpy.arange(bound_jianhe[2],bound_jianhe[3],size*pixel_size):
            x,y=round(i,5),round(j,5)
            lon,lat=geo2lonlat(dataset,x,y)
            point_lonlat=Point(lon,lat)
            if polygon_jianhe.contains(point_lonlat):
                points.append([lon,lat])
    result['jianhe']=points
    print(len(points))


    with open(r'E:\LIXINWEI\work\贵州省\codes\points.pkl','wb') as f:
        pickle.dump(result,f)


def generate_points_random():
    from matplotlib import pyplot
    path_shp_xixiu=r'H:\西秀区\行政区划\xixiu.shp'
    path_shp_jianhe=r'H:\剑河县\行政区划\jianhe.shp'
    num_xixiu=2000
    num_jianhe=2532

    shp_xixiu=gpd.read_file(path_shp_xixiu)
    shp_jianhe=gpd.read_file(path_shp_jianhe)

    polygon_xixiu=shp_xixiu['geometry']
    polygon_jianhe=shp_jianhe['geometry']



    bound_xixiu=[polygon_xixiu.bounds['minx'][0],polygon_xixiu.bounds['maxx'][0],polygon_xixiu.bounds['miny'][0],polygon_xixiu.bounds['maxy'][0]]
    bound_jianhe=[polygon_jianhe.bounds['minx'][0],polygon_jianhe.bounds['maxx'][0],polygon_jianhe.bounds['miny'][0],polygon_jianhe.bounds['maxy'][0]]

    result={'xixiu':None,'jianhe':None}

    #xixiu
    points=[]
    for i in range(num_xixiu):
        while True:
            x=round(random.uniform(bound_xixiu[0],bound_xixiu[1]),5)
            y=round(random.uniform(bound_xixiu[2],bound_xixiu[3]),5)
            point=Point(x,y)
            if polygon_xixiu.contains(point)[0]:
                break
        points.append([x,y])
    result['xixiu']=points




    #jianhe
    points=[]
    for i in range(num_jianhe):
        while True:
            x=round(random.uniform(bound_jianhe[0],bound_jianhe[1]),5)
            y=round(random.uniform(bound_jianhe[2],bound_jianhe[3]),5)
            point=Point(x,y)
            if polygon_jianhe.contains(point)[0]:
                break
        points.append([x,y])
    result['jianhe']=points

    with open(r'K:\points.pkl','wb') as f:
        pickle.dump(result,f)



def generate_thumb(path_src,path_dst):
    names=os.listdir(path_src)
    for name in names:
        n='.'.join(name.split('.')[:-1])
        data=gdal.Open(path_src+'\\'+name)
        clip_image=data.ReadAsArray().transpose(1,2,0)[:,:,[2,1,0]]
        if clip_image.max()>0:
            clip_image=clip_image/clip_image.max()*255
        Image.fromarray(clip_image.astype(numpy.uint8)).save(path_dst+'\\'+n+'.jpg')
                







if __name__=='__main__':
    
    from clip import clip_img
    from clip_label import clip_label_cons,clip_label_class
    
    #生成点位文件

    path_shp=r'D:\政区\贵州省县界.shp'   #贵州省县界shp文件
    size=2048                           #size
    pixel_size=0.65                     #像素大小   
    ref=r'D:\西秀区\2020影像\GF2_PMS2_E105.8_N26.0_20200826_L1A0005015595.tif'       #参考影像文件路径，提供投影坐标系，随便选一幅即可
    generate_points(path_shp,size,pixel_size,ref)   #生成一次点位文件（只用一次）
    
    
    '''
    #裁剪2021年影像
    
    id_county=1   #1表示剑河县   0表示西秀区
    year=2021     #年份
    path_pkl=r'E:\LIXINWEI\work\贵州省\codes\points.pkl'       #生成的点位文件
    path_src=r'D:\剑河县\光学影像2021'                          #待裁剪的文件夹
    path_dst=r'D:\剑河县\2021裁剪'                              #裁剪后保存的文件夹
    size=2048                                                  #size
    clip_img(id_county,year,path_pkl,path_src,path_dst,size,type='png')



    #裁剪2021年施工变化检测标签
    id_county=1   #1表示剑河县   0表示西秀区
    year=2021     #年份
    path_pkl=r'E:\LIXINWEI\work\贵州省\codes\points.pkl'       #生成的点位文件
    path_src=r''                          #待裁剪的文件夹
    path_dst=r''                              #裁剪后保存的文件夹
    size=2048                                                  #size
    clip_label_cons(id_county,year,path_pkl,path_src,path_dst,size)
   
    #裁剪2021年分类标签
    id_county=1   #1表示剑河县   0表示西秀区
    year=2021     #年份
    path_pkl=r'E:\LIXINWEI\work\贵州省\codes\points.pkl'       #生成的点位文件
    path_src=r''                          #待裁剪的文件夹
    path_dst=r''                              #裁剪后保存的文件夹
    size=2048                                                  #size
    clip_label_class(id_county,year,path_pkl,path_src,path_dst,size)
    '''
    