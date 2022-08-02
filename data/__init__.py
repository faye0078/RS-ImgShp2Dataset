from sympy import im
from data.GID_Vege_3bands import GIDVege3
from data.GID_Vege_4bands import GIDVege4
from data.GID_Vege_5bands import GIDVege5
from data.guangdong import Guangdong
from torch.utils.data import DataLoader, random_split
from data.concat import crop_patches
import sys
sys.path.append("../")
from utils.Path import get_train_path, get_predict_path
from data.transforms import (
    CentralCrop,
    Normalise,
    RandomCrop,
    RandomMirror,
    ResizeScale,
    ToTensor,
)
from torchvision import transforms
def make_train_loader(args, **kwargs):
    data_path = get_train_path(args.dataset)
    if args.dataset == 'GID-Vege3':
        Dataset = GIDVege3
    elif args.dataset == 'GID-Vege4':
        Dataset = GIDVege4
    elif args.dataset == 'GID-Vege5':
        Dataset = GIDVege5

    composed_trn = transforms.Compose(
        [
            RandomMirror(),
            RandomCrop(args.crop_size),
            ToTensor(),
        ]
    )
    composed_val = transforms.Compose(
        [

            CentralCrop(args.crop_size),
            ToTensor(),
        ]
    )
    composed_test = transforms.Compose(
        [
            CentralCrop(args.crop_size),
            ToTensor(),
        ])
    if args.nas == 'search':
        train_set = Dataset(stage="train",
                        data_file=data_path['nas_train_list'],
                        data_dir=data_path['dir'],
                        transform_trn=composed_trn, )
        val_set = Dataset(stage="val",
                        data_file=data_path['nas_val_list'],
                        data_dir=data_path['dir'],
                        transform_val=composed_val, )
    elif args.nas == 'train':
        train_set = Dataset(stage="train",
                        data_file=data_path['train_list'],
                        data_dir=data_path['dir'],
                        transform_trn=composed_trn,)
        val_set = Dataset(stage="val",
                        data_file=data_path['val_list'],
                        data_dir=data_path['dir'],
                        transform_val=composed_val,)
        test_set = Dataset(stage="test",
                        data_file=data_path['test_list'],
                        data_dir=data_path['dir'],
                        transform_test=composed_test,)
    else:
        raise Exception('nas param not set properly')

    n_examples = len(train_set)
    n_train = int(n_examples/2)
    train_set1, train_set2 = random_split(train_set, [n_train, n_examples - n_train])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    if args.nas == 'train':
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        print(" Created train set = {}, val set = {} examples, test set = {} examples".format(len(train_set), len(val_set), len(test_set)))
        return train_loader, val_loader, test_loader
    elif args.nas == 'search':
        print(" Created train setA = {} examples, train setB = {}, val set = {} examples".format(len(train_set1), len(train_set2), len(val_set)))
        return train_loader1, train_loader2, val_loader

def make_predict_split_loader(args, **kwargs):
    data_path = get_predict_path(args.dataset, args.mode)
    if args.dataset == 'GID-Vege3':
        Dataset = GIDVege3
    elif args.dataset == 'GID-Vege4':
        Dataset = GIDVege4
    elif args.dataset == 'GID-Vege5':
        Dataset = GIDVege5
    composed_test = transforms.Compose(
        [
            CentralCrop(args.crop_size),
            ToTensor(),
        ])

    test_set = Dataset(stage="test",
                    data_file=data_path['test_list'],
                    data_dir=data_path['dir'],
                    transform_test=composed_test,)

    print(" Created test set = {} examples".format(len(test_set)))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    return test_loader

def make_predict_concat_loader(args, **kwargs):
    data_path = get_predict_path(args.dataset, args.mode)
    if args.dataset == 'GID-Vege3':
        Dataset = GIDVege3
    elif args.dataset == 'GID-Vege4':
        Dataset = GIDVege4
    elif args.dataset == 'GID-Vege5':
        Dataset = GIDVege5
    elif args.dataset == 'Guangdong':
        Dataset = Guangdong
    composed_test = transforms.Compose(
        [
            # CentralCrop(args.crop_size),
            ToTensor(),
        ])

    test_set = Dataset(stage="predict",
                    data_file=data_path['predict_list'],
                    data_dir=data_path['dir'],
                    transform_test=composed_test,)

    print(" Created test set = {} examples".format(len(test_set)))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    test_dataloader = crop_patches(
        test_loader,
        args.origin_size,
        args.crop_size,
        args.stride
    )
    return test_dataloader