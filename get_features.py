import numpy
import h5py
import sys
import vigra
from scipy import stats

import Features

###############################
## Feature Calculation
###############################
##
## This code mostly stolen from get_superpixels.py
## See down below to see how to invoke Feature detection



# path to raw data file
d_path = "data/block00.h5"
d_h5_path = "volume/data"

# path to store features in
f_path = "data/features00.h5"

###############################
###############################


print "Starting to calculate segmentation..."

f = h5py.File("data/ws.h5")
ws = f["ws"].value
f.close()

f = h5py.File("data/block00.h5")
d = f[d_h5_path].value
f.close()

#####################################################
# Features 
#####################################################

print "Starting to Calculate Features on Supervoxels..."


proc = Features.Processing()

# currently we don't need scalers.
#proc.addScaler(Features.GaussianScaler(2.0))

proc.addChannelFeature(Features.MeanChannelValueFeature())

proc.addChannelGenerator(Features.TestChannelGenerator())

# Adds some channel generators
for scale in [1.0, 5.0]:
    proc.addChannelGenerator(Features.LaplaceChannelGenerator(scale))
    proc.addChannelGenerator(Features.GaussianGradientMagnitudeChannelGenerator(scale))
    proc.addChannelGenerator(Features.EVofGaussianHessianChannelGenerator(scale))

proc.addSupervoxelFeature(Features.SizeFeature())
proc.addSupervoxelFeature(Features.PCA())

features = proc.process(d, ws)


g = h5py.File(f_path, 'w')
g.create_dataset("features", data=features)
g.close()

print "Feature Array Shape: ", features.shape
