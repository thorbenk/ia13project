import numpy as np

class ChannelFeature:
    """
    Channel Generators create characteristics from the raw data. The aim of
    ChannelFeatures is to collect statistics for each supervoxel and channel.
    
    A ChannelFeature calculates from an (n,c)-shaped array statistics
    for each channel c and returns a vector of shape (numFeatures()*c,), where c is the number
    of channels provided and numFeatures() is the number of features that are calculated for
    each channel. N is the number of voxels that all belong to the same
    supervoxel.

    """

    def __init__(self):
        pass

    def name(self):
        return self.__class__.__name__

    def numFeatures(self, channels):
        """
        for the given number of channels, return here
        the total number of features that will be calculated.
        """
        raise NotImplementedError()

    def feature(self, voxels):
        """
        computes the features for the given voxels.
        voxels: (n,c)-shaped array.

        returns: (numFeatures()*c,) shaped array
        """
        raise NotImplementedError()



class MeanChannelValueFeature(ChannelFeature):
    """
    Computes for each channel the mean value over all voxels    
    """
    def features(self, voxels):
        return np.mean(voxels, axis=0)

    def numFeatures(self, channels):
        return channels


###############################
# TODO:
# Beside implementing the ChannelHistogramFeature 
# also add median, stddev and variance.

class ChannelHistogramFeature(ChannelFeature):
    """
    Computes for each channel a normalized histogram with given number of bins
    """

    def __init__(self, bins=10):
        self.bins = bins

    def numFeatures(self, channels):
        return  self.bins*channels

    def features(self, voxels):

        raise NotImplementedError()


# test the features
if __name__ == "__main__":

    #Test data (n, c), 2 voxels and 3 channels per voxel
    test = np.array([[1, 2, 3], [1, 2, 3]])
    
    mean = MeanChannelValueFeature()
    meanTest = mean.features(test)
    assert(hist.numFeatures(3) == 3)
    assert(meanTest.shape == (hist.numFeatures(1),))

#    #Test ChannelHistogramFeature
#    hist = ChannelHistogramFeature(10)
#    histTest = hist.features(test)
#    assert(hist.numFeatures(3) == 3*10)
#    assert(histTest.shape == (hist.numFeatures(3),))
    


    
