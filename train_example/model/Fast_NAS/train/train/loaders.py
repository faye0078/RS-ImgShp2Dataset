
import logging

from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from datasets import PascalCustomDataset as Dataset
from datasets import (
    CentralCrop,
    Normalise,
    RandomCrop,
    RandomMirror,
    ResizeScale,
    ToTensor,
)

def create_loaders(args):

    logger = logging.getLogger(__name__)
    composed_trn = transforms.Compose(
        [
            ResizeScale(
                args.resize_side[0],
                args.low_scale,
                args.high_scale,
                args.resize_longer_side,
            ),
            RandomMirror(),
            RandomCrop(args.crop_size[0]),
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    composed_val = transforms.Compose(
        [
            ResizeScale(args.val_resize_side, 1, 1, args.resize_longer_side),
            CentralCrop(args.val_crop_size),
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )

    trainset = Dataset(
        data_file=args.train_list,
        data_dir=args.train_dir,
        transform_trn=composed_trn,
        transform_val=composed_val,
    )
    do_search = False
    if args.train_list == args.val_list:
        do_search = True

        n_examples = len(trainset)
        n_train = int(n_examples * args.meta_train_prct / 100.0)
        trainset, valset = random_split(trainset, [n_train, n_examples - n_train])
    else:
        valset = Dataset(
            data_file=args.val_list,
            data_dir=args.val_dir,
            transform_trn=None,
            transform_val=composed_val,
        )
    logger.info(
        " Created train set = {} examples, val set = {} examples".format(
            len(trainset), len(valset)
        )
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size[0],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader