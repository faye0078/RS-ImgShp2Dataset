import numpy as np

LOWVEGE = ["0110", "0140", "0170", "0180", "0370", "0380", "0391", "0392", "0393", "03A1", "03A2", "03A3", "03A4", "03A9"]
HIGHVEGE = ["0131", "0132", "0133", "0311", "0312", "0313", "0321", "0322", "0323", "0330", "0340", "0350", "0360"]


def get_colormap():
    colormap= np.zeros((4,3), dtype=np.uint8)
    colormap[0] = [0,255,0]
    colormap[1] =  [255, 0, 0]
    colormap[2] =  [153,102,51]
    return colormap

def get_guiyang_labelmap(): # TODO: 改变规则
    labelmap = np.zeros((4,1), dtype=np.uint8)
    labelmap[0] = 0
    return labelmap