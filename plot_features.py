from pylab import *
from numpy import *

import get_config

import h5py

### features file and labels

features_path = "data/featuresk0.h5"
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
f = f_file["features"].value
c = f_file["channelInfo"].value
inf = f_file["featureInfo"].value
f_file.close()

l_file = h5py.File(labels_path)
l = l_file["ws"].value
l_file.close()

lslice = l[:,:,z_slide]


print l.shape
print f.shape

## some shortcuts
# number of supervoxels
nsv = f.shape[0]
nf = f.shape[1]


assert(len(l.shape) == 3)
assert(len(lslice.shape) == 2)
assert(len(f.shape) == 2)

print lslice.max(), "max lslice"

# create the colortable for labels
colortable = np.random.random((nsv+1, 3))

print colortable.shape
print f.shape

supervoxels = colortable[lslice]

# print the boundaries of the supervoxels
figure()
imshow(supervoxels)
colorbar()
title('Supervoxels')
savefig('./Features/plots/supervoxels.png', bbox_inches=0)

for i in range(72):
    figure("feature: "+str(i))
    feature = np.zeros((nsv+1))
    feature[1:] = f[:, i]
    output = feature[lslice]
    gray()
    imshow(output)
    colorbar()
    plotTitle = inf[i,1]
    picTitle = inf[i,1] + "_" + inf[i,0]
    channelNumber = inf[i,2]
    if channelNumber != str(len(c)):
       plotTitle += " of " + c[channelNumber, 0] + " of Scale " + c[channelNumber, 1]
       picTitle += "_" + c[channelNumber, 0] + "(" + c[channelNumber, 2] + ")_" + c[channelNumber, 1]
    title(plotTitle)
    savefig('./Features/plots/' + picTitle + '.png', bbox_inches=0)

