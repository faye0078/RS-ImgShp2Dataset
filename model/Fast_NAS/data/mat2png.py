import os
import scipy.io as io
from PIL import Image
import numpy as np
import collections

def SBD_label_mat2png():
    files = collections.defaultdict(list)
    palette = [255] * (256 * 3)
    palette[:(21 * 3)] = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128,
                          128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0,
                          64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128,
                          192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

    SBD_root_dir = 'C:/Users/Faye/Desktop/nas-segm-pytorch-master/src/sbd/dataset'

    # --- get all file list ---
    for split in ['train', 'val']:
        txt_file = os.path.join(SBD_root_dir, '%s.txt' % split)
        for idx in open(txt_file):
            idx = idx.strip()
            img_path = os.path.join(SBD_root_dir, 'img/%s.jpg' % idx)  # image file path
            lbl_mat_path = os.path.join(SBD_root_dir, 'inst/%s.mat' % idx)  # label mat file path
            files[split].append({
                'img_path': img_path,
                'lbl_mat_path': lbl_mat_path,
                'former_name': idx,
            })
    files['trainval'] = files['train'] + files['val']

    # --- convert mat to png ---
    cls_png_dir = os.path.join(SBD_root_dir, 'inst_png')
    if not os.path.exists(cls_png_dir):
        os.makedirs(cls_png_dir)

    for d in files['trainval']:
        # load label
        lbl_mat_path = d['lbl_mat_path']
        former_name = d['former_name']
        lbl_mat = io.loadmat(lbl_mat_path)
        lbl = lbl_mat['GTinst'][0]['Segmentation'][0].astype(np.uint8)
        lbl[lbl == 255] = -1
        lbl_img = Image.fromarray(lbl, 'P')
        lbl_img.putpalette(palette)
        lbl_img.save(os.path.join(cls_png_dir, '%s.png' % former_name))


if __name__ == "__main__":
    SBD_label_mat2png()