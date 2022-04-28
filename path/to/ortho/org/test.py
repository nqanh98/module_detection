from PIL import Image
import os
# import torch
import numpy as np

of = 'path/to/ortho/org/cluster_1_0001.npy'
# img = Image.open('path/to/ortho/org/cluster_1.tif')
# print(np.asarray(img).shape)
# print(np.asarray(img.convert('RGB')).shape)
basename = os.path.splitext(os.path.basename(of))[0].split('_')[0]
print(basename)
# original = np.expand_dims(np.asarray(img).transpose([2, 1, 0]), axis=0)
# print(original.shape)
