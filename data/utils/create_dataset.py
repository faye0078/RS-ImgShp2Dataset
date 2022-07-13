import numpy as np
import pandas as pd
import cv2
import os
['industrial_land', 'urban_residential', 'rural_residential',
 'traffic_land', 'paddy_field', 'irrigated_land',
  'dry_cropland', 'garden_land', 'arbor_woodland',
   'shrub_land', 'natural_grassland', 'artificial_grassland',
    'river', 'lake', 'pond', 'unknown']



def gid2Vege(label_dir):
    for dirpath, dirnames, filenames in os.walk(label_dir):
        for filename in filenames:
            label_path = os.path.join(dirpath, filename)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            target_label = np.zeros(label.shape)
            target_label[label == 4] = 0
            target_label[label == 5] = 0
            target_label[label == 10] = 0
            target_label[label == 11] = 0

            target_label[label == 8] = 1

            target_label[label == 0] = 2
            target_label[label == 1] = 2
            target_label[label == 2] = 2
            target_label[label == 3] = 2
            target_label[label == 6] = 2
            target_label[label == 12] = 2
            target_label[label == 13] = 2
            target_label[label == 14] = 2

            target_label[label == 255] = 255
            target_label[label == 7] = 255
            target_label[label == 9] = 255
            target_label = target_label.astype(np.uint)
            label_name = label_path.split('/')[-1]
            target_path = label_path.replace(label_name, '').replace('/label/', '/Vege_label/')
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            cv2.imwrite(target_path + label_name, target_label)

def changeFile():
    file = open("/media/dell/DATA/wy/LightRS/data/list/gid15_vege3_val.lst", "r", encoding='UTF-8')
    file_list = file.readlines()
    file_name = []
    for i in range(file_list.__len__()):
        a = str(file_list[i].replace('image/', 'image_NirRGB/')).replace('\n', '')
        file_name.append(a)
    df = pd.DataFrame(file_name, columns=['one'])
    df.to_csv('/media/dell/DATA/wy/LightRS/data/list/gid15_vege5_val.lst', columns=['one'], index=False, header=False)

    file.close()

changeFile()

# gid2Vege('/media/dell/DATA/wy/data/GID-15/512/label')