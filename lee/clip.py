#影像命名：县（0表示西秀，1表示剑河县）_序号(在points列表中的序号，从0开始)_同一位置的序号（同一位置可能有多张，标个序号，从0开始）_年份（2021之类的）_img
#变化标签命名：县（0表示西秀，1表示剑河县）_序号(在points列表中的序号，从0开始)_2020_2021_change
#分类标签命名：县（0表示西秀，1表示剑河县）_序号(在points列表中的序号，从0开始)_2021_label

from osgeo import gdal,osr
import pickle
import os
import numpy
from matplotlib import pyplot
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

def getAllImage_png(path):
    result=[]
    d=os.listdir(path)
    for i in d:
        if i.split('.')[-1] == 'png':
            result.append(i)
    return result

def contain(bound,point):
    if bound[0]<point[0]<bound[1] and bound[2]<point[1]<bound[3]:
        return True
    else:
        return False

def clip_img(county,year,path_pkl,path_src,path_dst,size=2048,type='tif'):
    bounds=[]
    dataset_list=[]
    if type=='tif':
        names=getAllImage_tif(path_src)
    else:
        names=getAllImage_png(path_src)
    for name in names:
        dataset=gdal.Open(path_src+'\\'+name)
        dataset_list.append(dataset)
        bounds.append(getImageBound(dataset))



    with open(path_pkl,'rb') as f:
        points_dict=pickle.load(f)
    if county==0:
        points=points_dict['xixiu']
    else:
        points=points_dict['jianhe']

    for j,point in enumerate(points,0):
        x,y=lonlat2geo(dataset_list[0],point[0],point[1])
        n=0
        for i,bound in enumerate(bounds,0):
            if contain(bound,(x,y)):
                p=geo2imagexy(dataset_list[i],x,y)
                if p[0]+size>dataset_list[i].RasterXSize or p[1]+size>dataset_list[i].RasterYSize:
                    continue
                if p[0]<0 or p[1]<0:
                    continue


                clip_image=dataset_list[i].ReadAsArray(p[0],p[1],size,size)

                clip_image_path=path_dst+'\\'+str(county)+'_'+str(j)+'_'+str(n)+'_'+str(year)+'_img.tif'
                clip_image_driver=gdal.GetDriverByName('GTiff')
                clip_image_dataset=clip_image_driver.Create(clip_image_path,size,size,4,gdal.GDT_Float32)
                clip_image_dataset.SetGeoTransform((x,dataset_list[i].GetGeoTransform()[1],0,y,0,dataset_list[i].GetGeoTransform()[5]))
                clip_image_dataset.SetProjection(dataset_list[i].GetProjection())
                for k in range(4):
                    clip_image_dataset.GetRasterBand(k+1).WriteArray(clip_image[k,:,:])
                clip_image_dataset.FlushCache()
                clip_image_dataset=None

                n+=1

def clip_img_split(path_src,path_dst):
    '''
    裁剪为512，并去掉地理坐标
    '''
    names=os.listdir(path_src)

    for name in names:
        data=gdal.Open(path_src+'\\'+name)
        n='.'.join(name.split('.')[:-1])
        for i in range(0,2048,512):
            for j in range(0,2048,512):
                clip_image=data.ReadAsArray(i,j,512,512).transpose(1,2,0)[:,:,[2,1,0]]
                
                clip_image=clip_image/clip_image.max()*255
                Image.fromarray(clip_image.astype(numpy.uint8)).save(path_dst+'\\'+n+'_'+str(i)+'_'+str(j)+'.tif')
                

if __name__=='__main__':
    path_src=r'D:\西秀区\2020裁剪'
    path_dst=r'F:\LIXINWEI\gy_data\test'

    clip_img_split(path_src,path_dst)
    '''
    id_county=0
    year=2020
    path_pkl=r'E:\LIXINWEI\work\贵州省\codes\points.pkl'
    path_src=r'D:\西秀区\2020影像'
    path_dst=r'D:\西秀区\2020裁剪'
    size=2048
    clip_img(id_county,year,path_pkl,path_src,path_dst,size)

    id_county=0
    year=2021
    path_pkl=r'E:\LIXINWEI\work\贵州省\codes\points.pkl'
    path_src=r'D:\西秀区\2021重采样'
    path_dst=r'D:\西秀区\2021裁剪'
    size=2048
    clip_img(id_county,year,path_pkl,path_src,path_dst,size)


    id_county=1
    year=2020
    path_pkl=r'E:\LIXINWEI\work\贵州省\codes\points.pkl'
    path_src=r'D:\剑河县\2020影像'
    path_dst=r'D:\剑河县\2020裁剪'
    size=2048
    clip_img(id_county,year,path_pkl,path_src,path_dst,size)
    '''
