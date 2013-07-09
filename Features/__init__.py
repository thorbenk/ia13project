"""
Feature Module.
Create Features for given Supervoxels.

See Processing.py for further information
"""

from Processing import Processing
from Scalers import *
from ChannelFeatures import *
from ChannelGenerators import *
from SupervoxelFeatures import *


def getRichFeatureSet():
    raise NotImplementedError()

def getMinimalFeatureSet():
    """
    Returns a Processing Instance with minimal feature set
    """
    proc = Processing()

    # channel features
    proc.addChannelFeature(MeanChannelValueFeature())
    #proc.addChannelFeature(ChannelHistogramFeature())
    
    # channel generators
    scale = 1.0
    proc.addChannelGenerator(LaplaceChannelGenerator(scale))
    proc.addChannelGenerator(GaussianGradientMagnitudeChannelGenerator(scale))
    proc.addChannelGenerator(EVofGaussianHessianChannelGenerator(scale))

    # supervoxel features
    proc.addSupervoxelFeature(SizeFeature())

    return proc



def getTestingFeatureSet():
    raise NotImplementedError()
