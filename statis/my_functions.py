from osgeo import gdal
from collections import Counter
import numpy as np
def caculate_distribution(label_path):
    label_dataset = gdal.Open(label_path)
    
    label_array = label_dataset.ReadAsArray()
    dis = {}
    for i in np.unique(label_array):
        dis[str(int(i))] = sum(sum(label_array==i))
    
    return Counter(dis)