from pyside import *
from numpy import *

import h5py

### features file and labels

features_path = "data/features00.h5"
features_h5_path = "features"

labels_path = "data/ws.h5"
labels_h5_path = "ws"

### display options

# the slide to select to show feature values for
z_slide = 10

# select the features you want to display
features = [0, 1, 2, 3]

#######################################################


### import features

f_file = h5py.File(features_path)
f = f_file["features"]
f_file.close()

l_file = h5py.File(labels_path)
l = l_file["ws"]
l_file.close()


dataslice = l[:,:,z_slide]

assert(len(data.shape) == 2)




