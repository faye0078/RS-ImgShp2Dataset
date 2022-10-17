import os
import glob
from osgeo import gdal
from osgeo import ogr
os.environ['SHAPE_ENCODING'] = 'uft-8'
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")

lowVege = ["0110", "0140", "0170", "0180", "0370", "0380", "0391", "0392", "0393", "03A1", "03A2", "03A3", "03A4", "03A9"]
highVege = ["0131", "0132", "0133", "0311", "0312", "0313", "0321", "0322", "0323", "0330", "0340", "0350", "0360"]
building = []
lowVegeSQL = "CC = '0110' OR CC = '0140' OR CC = '0170' OR CC = '0180' OR CC = '0370' OR CC = '0380' OR CC = '0391'" \
             " OR CC = '0392' OR CC = '0393' OR CC = '03A1' OR CC = '03A2' OR CC = '03A3' OR CC = '03A4' OR CC = '03A9'"
highVegeSQL = "CC = '0131' OR CC = '0132' OR CC = '0133' OR CC = '0311' OR CC = '0312' OR CC = '0313' OR CC = '0321' OR CC = '0322'" \
              " OR CC = '0323' OR CC = '0330' OR CC = '0340' OR CC = '0350' OR CC = '0360'"
buildingSQL = ""
# trueSQL highVegeSQL = "SELECT * FROM %s WHERE CC = '0323'"

def select(filename, output_file, SQL):
    ogr.RegisterAll()
    datasource = ogr.Open(filename)
    layer = datasource.GetLayer(0)
    # high Vege
    layer.SetAttributeFilter(SQL)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.access(output_file, os.F_OK):
        driver.DeleteDataSource(output_file)
    newds = driver.CreateDataSource(output_file)
    pt_layer = newds.CopyLayer(layer, "abcd")
    newds.Destroy()

if __name__ == "__main__":
    # config: input dir, output dir, and SQL sentence
    data_dir = "J:/GuangdongSHP/2019/"
    out_dir = "J:/GuangdongSHP/lowVege/"
    # option: lowVegeSQL, highVegeSQL, buildingSQL
    SQL = lowVegeSQL

    file_list = glob.glob(('{}*.shp'.format(data_dir)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, file_name in enumerate(file_list):
        print("{}/{}".format(str(i), str(len(file_list))))
        file_name = file_name.replace("\\", "/")
        output_file = out_dir + file_name.split("/")[-1]
        select(file_name, output_file, SQL)
