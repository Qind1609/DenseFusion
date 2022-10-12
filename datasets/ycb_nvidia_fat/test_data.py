import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import sys
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio

np.set_printoptions(threshold=sys.maxsize)
path = "/media/qind/Data/QIND-DATA/fat/mixed/temple_3/"
input_file = Image.open(path+'001026.left.depth.png')


input_file = np.array(input_file)*(255/(np.max(input_file)-np.min(input_file)))

data1 = Image.fromarray(input_file)
data1.show()
#input_file = ma.getmaskarray(ma.masked_not_equal(input_file, 0))
with open('text.txt','w') as f:
    input_file = str(input_file)
    f.write(input_file)
    f.close()