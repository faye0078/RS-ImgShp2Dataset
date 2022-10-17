from collections import OrderedDict
def get_train_path(dataset):
    if dataset == 'GID-Vege3':
        Path = OrderedDict()
        Path['dir'] = "/media/dell/DATA/wy/data/GID-15/512/"
        Path['train_list'] = "/media/dell/DATA/wy/LightRS/data/list/split/gid15_vege3_train.lst"
        Path['val_list'] = "/media/dell/DATA/wy/LightRS/data/list/split/gid15_vege3_val.lst"
        Path['test_list'] = "/media/dell/DATA/wy/LightRS/data/list/split/gid15_vege3_val.lst"

        # Path['nas_train_list'] = "./data/lists/mini_hps_train.lst"
        # Path['nas_val_list'] = "./data/lists/mini_hps_val.lst"
    elif dataset == 'GID-Vege4' or dataset == 'GID-Vege5':
        Path = OrderedDict()
        Path['dir'] = "/media/dell/DATA/wy/data/GID-15/512/"
        Path['train_list'] = "/media/dell/DATA/wy/LightRS/data/list/split/gid15_vege4_train.lst"
        Path['val_list'] = "/media/dell/DATA/wy/LightRS/data/list/split/gid15_vege4_val.lst"
        Path['test_list'] = "/media/dell/DATA/wy/LightRS/data/list/split/gid15_vege4_val.lst"
        
        # Path['nas_train_list'] = "./data/lists/GID/mini_rs_train.lst"
        # Path['nas_val_list'] = "./data/lists/GID/mini_rs_val.lst"
    elif dataset == 'Guangdong_train':
        Path = OrderedDict()
        Path['dir'] = ""
        Path['train_list'] = "F:/WHU_WY/LightRS/data/list/split/guangdong_train.lst"
        # Path['val_list'] = "F:/WHU_WY/LightRS/data/list/split/guangdong_val.lst"
        # Path['test_list'] = "F:/WHU_WY/LightRS/data/list/split/guangdong_val.lst"
        Path['val_list'] = "F:/WHU_WY/LightRS/data/list/split/guangdong_val.lst"
        Path['test_list'] = "F:/WHU_WY/LightRS/data/list/split/guangdong_test.lst"
    elif dataset == 'Guangzhou':
        Path = OrderedDict()
        Path["wangyu"] = 'wangyu'
    return Path

def get_predict_path(dataset, mode):
    if dataset == 'GID-Vege3':
        Path = OrderedDict()
        if mode == 'split':
            Path['dir'] = "/media/dell/DATA/wy/data/GID-15/512/"
        elif mode == 'concat':
            Path['dir'] = "/media/dell/DATA/wy/data/GID-15/GID/"
        Path['train_list'] = "/media/dell/DATA/wy/LightRS/data/list/{}/gid15_vege3_train.lst".format(mode)
        Path['val_list'] = "/media/dell/DATA/wy/LightRS/data/list/{}/gid15_vege3_val.lst".format(mode)
        Path['test_list'] = "/media/dell/DATA/wy/LightRS/data/list/{}/gid15_vege3_val.lst".format(mode)

        # Path['nas_train_list'] = "./data/lists/mini_hps_train.lst"
        # Path['nas_val_list'] = "./data/lists/mini_hps_val.lst"
    elif dataset == 'GID-Vege4' or dataset == 'GID-Vege5':
        Path = OrderedDict()
        if mode == 'split':
            Path['dir'] = "/media/dell/DATA/wy/data/GID-15/512/"
        elif mode == 'concat':
            Path['dir'] = "/media/dell/DATA/wy/data/GID-15/GID/"
        Path['dir'] = "/media/dell/DATA/wy/data/GID-15/512/"
        Path['train_list'] = "/media/dell/DATA/wy/LightRS/data/list/{}/gid15_vege4_train.lst".format(mode)
        Path['val_list'] = "/media/dell/DATA/wy/LightRS/data/list/{}/gid15_vege4_val.lst".format(mode)
        Path['test_list'] = "/media/dell/DATA/wy/LightRS/data/list/{}/gid15_vege4_val.lst".format(mode)
        
        # Path['nas_train_list'] = "./data/lists/GID/mini_rs_train.lst"
        # Path['nas_val_list'] = "./data/lists/GID/mini_rs_val.lst"
    elif dataset == 'Guangdong':
        Path = OrderedDict()
        Path['dir'] = "F:/WHU_WY/LightRS/data/"
        Path["predict_list"] = "./data/list/concat/guangdong_predict.lst" # TODO:路径

    return Path
#
# file/20DEC01030058-M2AS-013317249080_04_P003_FUS.tif
# file/20DEC01030059-M2AS-013317249080_04_P004_FUS.tif
# file/20DEC06031601-M2AS-013317249080_05_P001_FUS.tif
# file/20JAN29031211-M2AS-012339292060_01_P005_FUS.tif