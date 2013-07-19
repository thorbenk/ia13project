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

def computeSuperpixelFeatures(blockName, slicing):
    # path to raw data file
    d_path = "data/%s.h5" % blockName
    d_h5_path = "volume/data"

    # path to store features in
    f_path = "data/features%s.h5" % blockName[-3:-1]

    ###############################
    ###############################


    print "Starting to calculate segmentation..."

    f = h5py.File("data/ws.h5")
    ws = f["ws"].value
    f.close()

    f = h5py.File("data/%s.h5" % blockName)
    d = f[d_h5_path][slicing]
    f.close()

    #####################################################
    # Features 
    #####################################################

    print "Starting to Calculate Features on Supervoxels..."


    proc = Features.Processing()

    # currently we don't need scalers.
    #proc.addScaler(Features.GaussianScaler(2.0))

    proc.addChannelFeature(Features.MeanChannelFeature())
    proc.addChannelFeature(Features.MedianChannelFeature())
    proc.addChannelFeature(Features.StdDeviationChannelFeature())
    proc.addChannelFeature(Features.VarianceChannelFeature())
    # histogram
    proc.addChannelFeature(Features.ChannelHistogramFeature(5))
    proc.addChannelFeature(Features.ChannelHistogramFeature(10))
    proc.addChannelFeature(Features.ChannelHistogramFeature(20))

    # Adds some channel generators
    for scale in [1.0, 5.0]:
        proc.addChannelGenerator(Features.SmoothImageChannelGenerator(scale))
        proc.addChannelGenerator(Features.LaplaceChannelGenerator(scale))
        proc.addChannelGenerator(Features.GaussianGradientMagnitudeChannelGenerator(scale))
        proc.addChannelGenerator(Features.EVofGaussianHessianChannelGenerator(scale))
        proc.addChannelGenerator(Features.EVofStructureTensorChannelGenerator(scale))

    proc.addSupervoxelFeature(Features.SizeFeature())
    proc.addSupervoxelFeature(Features.PCA())

    features, featureInfo, channelInfo = proc.process(d, ws)

    g = h5py.File(f_path, 'w')
    g.create_dataset("features", data=features)
    g.create_dataset("channelInfo", data=channelInfo)
    g.create_dataset("featureInfo", data=featureInfo)
    g.close()

    print "Feature Array Shape: ", features.shape

if __name__ == '__main__':
    from get_config import small_slicing
    computeSuperpixelFeatures("block00", small_slicing)
