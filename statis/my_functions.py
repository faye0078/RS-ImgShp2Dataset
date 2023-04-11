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


c = {'4': 11461364379,
     '5': 5913987322,
     '1': 5023859192,
     '0': 1978150491,
     '2': 880303201,
     '11': 648322934,
     '6': 596219288,
     '14': 435017831,
     '16': 413338656,
     '7': 395991816,
     '13': 312444499, 
     '12': 290279835, 
     '10': 286261040, 
     '9': 226447717, 
     '8': 209486391, 
     '3': 193868098, 
     '15': 86580428}

# sum = sum(c.values())
# for i in c.keys():
#     c[i] = c[i]/sum
# sort by key
c = dict(sorted(c.items(), key=lambda item: item[0]))
print(c)