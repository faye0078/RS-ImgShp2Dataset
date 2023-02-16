#影像命名：县（0表示西秀，1表示剑河县）_序号(在points列表中的序号，从0开始)_同一位置的序号（同一位置可能有多张，标个序号，从0开始）_年份（2021之类的）_img
#施工标签命名：县（0表示西秀，1表示剑河县）_序号(在points列表中的序号，从0开始)_年份（2021之类的）_conslabel
#分类标签命名：县（0表示西秀，1表示剑河县）_序号(在points列表中的序号，从0开始)_2021_classlabel

from osgeo import gdal,osr
import pickle
import os
import numpy

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

def geo2imagexy(dataset,x,y):
    trans=dataset.GetGeoTransform()
    a=numpy.array([[trans[1],trans[2]],[trans[4],trans[5]]])
    b=numpy.array([x-trans[0],y-trans[3]])
    result=numpy.linalg.solve(a,b)
    return int(result[0]),int(result[1])

def getImageBound(dataset):
    #[minx.maxx,miny,maxy]
    trans=dataset.GetGeoTransform()
    img_width,img_height=dataset.RasterXSize,dataset.RasterYSize
    result=[min(trans[0],trans[0]+trans[1]*img_width),max(trans[0],trans[0]+trans[1]*img_width),min(trans[3],trans[3]+trans[5]*img_height),max(trans[3],trans[3]+trans[5]*img_height)]
    return result

def getAllImage_tif(path):
    result=[]
    d=os.listdir(path)
    for i in d:
        if i.split('.')[-1] == 'tif':
            result.append(i)
    return result

def contain(bound,point):
    if bound[0]<point[0]<bound[1] and bound[2]<point[1]<bound[3]:
        return True
    else:
        return False


def clip_label_cons(county,year,path_pkl,path_src,path_dst,size=2048):
    dataset=gdal.Open(path_src)
    bound=getImageBound(dataset)


    with open(path_pkl,'rb') as f:
        points_dict=pickle.load(f)
    if county==0:
        points=points_dict['xixiu']
    else:
        points=points_dict['jianhe']

    for j,point in enumerate(points,0):
        x,y=lonlat2geo(dataset,point[0],point[1])
        if contain(bound,(x,y)):
            p=geo2imagexy(dataset,x,y)
            if p[0]+size>dataset.RasterXSize or p[1]+size>dataset.RasterYSize:
                continue
            clip_image=dataset.ReadAsArray(p[0],p[1],size,size)

            clip_image_path=path_dst+'\\'+str(county)+'_'+str(j)+'_'+str(year)+'_conslabel.tif'
            clip_image_driver=gdal.GetDriverByName('GTiff')
            clip_image_dataset=clip_image_driver.Create(clip_image_path,size,size,1,gdal.GDT_Float32)
            clip_image_dataset.SetGeoTransform((x,dataset.GetGeoTransform()[1],0,y,0,dataset.GetGeoTransform()[5]))
            clip_image_dataset.SetProjection(dataset.GetProjection())
            clip_image_dataset.GetRasterBand(1).WriteArray(clip_image)
            clip_image_dataset.FlushCache()
            clip_image_dataset=None

def clip_label_class(county,year,path_pkl,path_src,path_dst,size=2048):
    dataset=gdal.Open(path_src)
    bound=getImageBound(dataset)


    with open(path_pkl,'rb') as f:
        points_dict=pickle.load(f)
    if county==0:
        points=points_dict['xixiu']
    else:
        points=points_dict['jianhe']

    for j,point in enumerate(points,0):
        x,y=lonlat2geo(dataset,point[0],point[1])
        if contain(bound,(x,y)):
            p=geo2imagexy(dataset,x,y)
            if p[0]+size>dataset.RasterXSize or p[1]+size>dataset.RasterYSize:
                continue
            clip_image=dataset.ReadAsArray(p[0],p[1],size,size)

            clip_image_path=path_dst+'\\'+str(county)+'_'+str(j)+'_'+str(year)+'_classlabel.tif'
            clip_image_driver=gdal.GetDriverByName('GTiff')
            clip_image_dataset=clip_image_driver.Create(clip_image_path,size,size,1,gdal.GDT_Float32)
            clip_image_dataset.SetGeoTransform((x,dataset.GetGeoTransform()[1],0,y,0,dataset.GetGeoTransform()[5]))
            clip_image_dataset.SetProjection(dataset.GetProjection())
            clip_image_dataset.GetRasterBand(1).WriteArray(clip_image)
            clip_image_dataset.FlushCache()
            clip_image_dataset=None


if __name__=='__main__':
    from matplotlib import pyplot

    id_county=1
    year=2021
    path_pkl=r'K:\points.pkl'
    path_src=r'H:\剑河县\2021标签\label_2021.tif'
    path_dst=r'H:\剑河县\2021标签裁剪\分类'
    size=2048
    clip_label_class(id_county,year,path_pkl,path_src,path_dst,size)

    id_county=1
    year=2021
    path_pkl=r'K:\points.pkl'
    path_src=r'H:\剑河县\2021标签\label_2021_cons.tif'
    path_dst=r'H:\剑河县\2021标签裁剪\施工'
    size=2048
    clip_label_cons(id_county,year,path_pkl,path_src,path_dst,size)

    id_county=1
    year=2020
    path_pkl=r'K:\points.pkl'
    path_src=r'H:\剑河县\2020标签\label_2020.tif'
    path_dst=r'H:\剑河县\2020标签裁剪\施工'
    size=2048
    clip_label_cons(id_county,year,path_pkl,path_src,path_dst,size)
