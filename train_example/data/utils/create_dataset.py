import numpy as np
import pandas as pd
import cv2
import os
import glob
['industrial_land', 'urban_residential', 'rural_residential',
 'traffic_land', 'paddy_field', 'irrigated_land',
  'dry_cropland', 'garden_land', 'arbor_woodland',
   'shrub_land', 'natural_grassland', 'artificial_grassland',
    'river', 'lake', 'pond', 'unknown']



def gid2Vege(label_dir):
    for dirpath, dirnames, filenames in os.walk(label_dir):
        for filename in filenames:
            label_path = os.path.join(dirpath, filename)
            if "tif" in label_path:
                continue
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label[label==0] = 255
            label = label - 1
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

            target_label[label > 15] = 255
            target_label[label == 7] = 255
            target_label[label == 9] = 255
            target_label = target_label.astype(np.uint)
            label_name = label_path.split('/')[-1]
            target_path = label_path.replace(label_name, '').replace('/label/', '/Vege_label/')
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            cv2.imwrite(target_path + label_name, target_label)

def make_concat_lst(data_path):
    imgs = glob.glob(('{}*.tif'.format(data_path)))
    file = []
    for img in imgs:
        img = img.replace('/media/dell/DATA/wy/data/gid-15/GID/img_dir/val/', 'image_NirRGB/')
        label = img.replace('image_NirRGB/', 'Vege_label/').replace('.tif', '_15label.png')
        file.append(img + '\t' + label)
    df = pd.DataFrame(file, columns=['one'])
    # df = df.sample(frac=0.05, random_state=1)
    df.to_csv("/media/dell/DATA/wy/LightRS/data/list/concat/gid15_vege4_val.lst", columns=['one'], index=False, header=False)

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

def make_predict_list(data_path, result_path):
    imgs = glob.glob(('{}*.tif'.format(data_path)))
    file = []
    for img in imgs:
        img = img.replace('I:/LightRS/data/', '').replace('\\', '/')
        file.append(img)
    df = pd.DataFrame(file, columns=['one'])
    df.to_csv(result_path, columns=['one'], index=False, header=False)

def make_guangdong_list(dir):
    file_list = []
    i = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        if len(filenames) != 0:
            for filename in filenames:
                name = os.path.join(dirpath, filename)
                name = name.replace("\\", "/")
                # label_name = name.replace('image_', '_label_').replace('_img', '_label').replace('.tif', '.png')
                label_name = name.replace('image', 'label').replace('img', 'label')
                image = cv2.imread(name)
                if len(np.where(np.all(image == [0, 0, 0], axis=-1))[0]) == 0:
                    file_list.append(name + '\t' + label_name)
                    i = i + 1
                    print(i)
    df = pd.DataFrame(file_list, columns=['one'])
    df.to_csv("F:/WHU_WY/LightRS/data/list/split/guangdong_val.lst", columns=['one'], index=False, header=False)

make_guangdong_list("F:/WHU_WY/data/512/image/val/")
# changeFile()
# gid2Vege('/media/dell/DATA/wy/data/GID-15/GID/label')
# make_concat_lst("/media/dell/DATA/wy/data/gid-15/GID/img_dir/val/")
# make_predict_list(data_path="I:/LightRS/data/file/", result_path="I:/LightRS/data/list/concat/guangdong_predict.lst")