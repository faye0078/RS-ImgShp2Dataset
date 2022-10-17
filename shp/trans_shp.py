from osgeo import ogr
import glob
driver = ogr.GetDriverByName("ESRI Shapefile")
lowVege = ["0110", "0140", "0170", "0180", "0370", "0380", "0391", "0392", "0393", "03A1", "03A2", "03A3", "03A4", "03A9"]
highVege = ["0131", "0132", "0133", "0311", "0312", "0313", "0321", "0322", "0323", "0330", "0340", "0350", "0360"]
unknow = []
def trans_shp(fn):
    dataSource = driver.Open(fn, 1)
    layer = dataSource.GetLayer()
    feature = layer.GetNextFeature()
    newField = ogr.FieldDefn('Vege', ogr.OFTInteger)
    layer.CreateField(newField)

    while feature:
        CC = feature.GetField('CC')
        if CC in lowVege:
            feature.SetField('Vege', 0)
        elif CC in highVege:
            feature.SetField('Vege', 1)
        elif CC in unknow:
            feature.SetField('Vege', 3)
        else:
            feature.SetField('Vege', 2)
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    return
if __name__ == "__main__":
    # data_dir = "J:/GuangdongSHP/selsectSHP/"
    # file_list = glob.glob(('{}*.shp'.format(data_dir)))
    # for file in file_list:
    #     trans_shp(file)
    trans_shp("J:/GuangdongSHP/splitSHP/2/merged.shp")
