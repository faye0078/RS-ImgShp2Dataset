import os, random
import numpy as np
from PIL import Image

from torch.utils import data
# from dataloaders import custom_transforms as tr
import rasterio
from rasterio.enums import Resampling
import torch
import torchvision.transforms.functional as TF

def twoTrainSeg(args, root=Path.db_root_dir('gid')):
    images_base = os.path.join(root, 'all', 'train')
    train_files = [os.path.join(looproot, filename) for looproot, _, filenames in os.walk(images_base)
                   for filename in filenames if filename.endswith('.tif')]
    number_images = len(train_files)
    permuted_indices_ls = np.random.permutation(number_images)
    indices_1 = permuted_indices_ls[: int(0.5 * number_images) + 1]
    indices_2 = permuted_indices_ls[int(0.5 * number_images):]
    if len(indices_1) % 2 != 0 or len(indices_2) % 2 != 0:
        raise Exception('indices lists need to be even numbers for batch norm')
    return gidSegmentation(args, split='train', indices_for_split=indices_1), gidSegmentation(args, split='train', indices_for_split=indices_2)

def get_gid_labels():
    return np.array([
        [255,0,0],    #buildup
        [0,255,0],   #farmland
        [0,255,255],  #forest
        [255,255,0],  #meadow
        [0,0,255] ])  #water

class gidSegmentation(data.Dataset):
    NUM_CLASSES = 5
    CLASSES = ['buildup', 'farmland', 'forest', 'meadow', 'water']

    def __init__(self, args, root=Path.db_root_dir('gid'), split="train", indices_for_split=None):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.mean = (0.4965,0.3704,0.3900,0.3623)
         # (104.00698793, 116.66876762, 122.67891434)
        self.std = (0.2412,0.2297,0.2221,0.2188)
        self.crop = self.args.crop_size
        if split.startswith('re'):
            self.images_base = os.path.join(self.root, self.split[2:], 'image')
            self.annotations_base = os.path.join(self.root, self.split[2:], 'label')
        else:
            self.images_base = os.path.join(self.root, self.split, 'image')
            self.annotations_base = os.path.join(self.root, self.split, 'label')

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.tif')

        if indices_for_split is not None:
            self.files[split] = np.array(self.files[split])[indices_for_split].tolist()

        # self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['buildup', 'farmland', 'forest', 'meadow', 'water']

        self.ignore_index = 255
        # self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))
        # self.transform = self.get_transform()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path.replace('image','label').replace('.tif','_label.tif').replace('/768','/512').replace('scale3','scale2')
        # lbl_path = img_path.replace('image','label').replace('.tif','_label.tif')
        name = os.path.basename(img_path)
        # _img = Image.open(img_path).convert('RGB')
        # _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        # _tmp = self.encode_segmap(_tmp)
        # _target = Image.fromarray(_tmp)

        #TODO during retrain， add multiscale resize in my_transforms LINE234

        with rasterio.open(img_path) as image:
            # _img = image.read(out_shape=(image.count, self.args.resize, self.args.resize), resampling=Resampling.bilinear).astype(np.float32).transpose(1,2,0)
            _img = image.read().astype(np.float32).transpose(1,2,0)
        with rasterio.open(lbl_path) as label:
            # _tmp = label.read(out_shape=(self.args.resize, self.args.resize, label.count), resampling=Resampling.nearest)
            _tmp = label.read()
        _tmp = np.array(_tmp).astype(np.uint8).transpose(1,2,0)
        _tmp = self.encode_segmap(_tmp).astype(np.float32)
        _img /= 255.0
        _img -= self.mean
        _img /= self.std
        _img = _img.transpose(2,0,1)
        _img = torch.from_numpy(_img).float()
        _tmp = torch.from_numpy(_tmp).float()

        if 'train' in self.split:
            a = random.random()
            if a < 0.5:
                _img = TF.hflip(_img)
                _tmp = TF.hflip(_tmp)
            b = random.random()
            if b < 0.5:
                _img = TF.vflip(_img)
                _tmp = TF.vflip(_tmp)
            c = random.random()
            if c < 0.5:
                _img = TF.rotate(_img, 90)
                _tmp = TF.rotate(_tmp.unsqueeze(0), 90).squeeze()

            # if 'dsp' not in self.root:
            #     x1 = random.randint(0, _img.shape[1] - self.crop)
            #     y1 = random.randint(0, _img.shape[2] - self.crop)
            #     _img = TF.crop(_img,x1,y1,self.crop,self.crop)
            #     _tmp = TF.crop(_tmp,x1,y1,self.crop,self.crop)
        # flip
        # randomcrop local only

        sample = {'image': _img, 'label': _tmp, 'name': name}

        # return self.transform(sample)
        return sample

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = np.uint8(mask)
        # label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        label_mask = 255 * np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(get_gid_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = np.uint8(label_mask)
        return label_mask

    # def encode_segmap(self, mask):
    #     # Put all void classes to zero
    #     for _voidc in self.void_classes:
    #         mask[mask == _voidc] = self.ignore_index
    #     for _validc in self.valid_classes:
    #         mask[mask == _validc] = self.class_map[_validc]
    #     return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def get_transform(self):
        if self.split == 'train':
            return tr.transform_tr(self.args, self.mean, self.std)
        elif self.split == 'val':
            return tr.transform_val(self.args, self.mean, self.std)
        elif self.split == 'test':
            return tr.transform_ts(self.args, self.mean, self.std)
        elif self.split == 'retrain':
            return tr.transform_retr(self.args, self.mean, self.std)
        elif self.split == 'reval':
            return tr.transform_reval(self.args, self.mean, self.std)


if __name__ == '__main__':
    from dataloaders.dataloader_utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.resize = 513
    args.base_size = 513
    args.crop_size = 513

    gid_train = gidSegmentation(args, split='retrain')

    dataloader = DataLoader(gid_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='gid')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
