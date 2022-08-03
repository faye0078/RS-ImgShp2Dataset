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
        Path['dir'] = "/media/dell/DATA/wy/data/"
        Path["predict_list"] = "/media/dell/DATA/wy/LightRS/data/list/concat/guangdong_predict.lst" # TODO:路径

    return Path