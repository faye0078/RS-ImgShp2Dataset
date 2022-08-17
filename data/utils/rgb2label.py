import numpy as np
from PIL import Image
from PIL import ImageFile
import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
def get_vege_labels():
    return np.array([
        [0,255,0],    # 0: 低植被
        [255,0,0],   #1：高植被
        [2,2,2],])  #2：非植被
 
def rgb2label(dir_path):
    imgs = glob.glob('{}*.png'.format(dir_path))
    for filename in imgs:
        if filename == "/media/dell/DATA/wy/data/Guangdong/dataset/origin_label/label_scene1.png":
            img = np.array(Image.open(filename))
            Image.fromarray(img).save(filename.replace('.png', '_trans.png'))
            continue
        img = np.array(Image.open(filename))
        label_mask = 255 * np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(get_vege_labels()):
            label_mask[np.where(np.all(img == label, axis=-1))[:2]] = ii
        label_mask = np.uint8(label_mask)
        Image.fromarray(label_mask).save(filename.replace('.png', '_trans.png'))

if __name__ == "__main__":
    dir_path = "/media/dell/DATA/wy/data/Guangdong/dataset/origin_label/"
    rgb2label(dir_path)