import numpy as np

class ChannelFeature:
    """
    A ChannelFeature calculates from a (n,c)-shaped array interesting statistics
    for each channel c and returns a vector of shape (a*c,), where n is the number
    of voxels provided and a is the number of features that are calculated for
    each channel.
    """

    def __init__(self):
        pass

    def name(self):
        return self.__class__.__name__

    def numFeatures(self, channels):
        """
        for the given number of channels, 
        returns the number of features this class will
        compute
        """
        raise NotImplementedError()

    def feature(self, voxels):
        """
        computes the features for the given voxels
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


class ChannelHistogramFeature(ChannelFeature):
    """
    Computes for each channel a normalized histogram with given number of bins
    """

    def __init__(self, bins=10):
        self.bins = bins

    def numFeatures(self, channels):
        return  self.bins*channels

    def features(self, voxels):
        return np.array(np.ones(3*10))


if __name__ == "__main__":

    #Test data
    test = np.array([[1, 2, 3], [1, 2, 3]])
    
    
    #Test MeanChannelValueFeature
    mean = MeanChannelValueFeature()
    meanTest = mean.features(test)
    assert(meanTest.shape == (3,))

    #Test ChannelHistogramFeature
    hist = ChannelHistogramFeature(10)
    histTest = hist.features(test)
    assert(histTest.shape == (3*10,))
    


    
