import numpy as np

class ChannelFeature:
    """
    Channel Generators create characteristics from the raw data. The aim of
    ChannelFeatures is to collect statistics for each supervoxel and channel.
    
    A ChannelFeature calculates from an (n,c)-shaped array statistics
    for each channel c and returns a vector of shape (numFeatures()*c,), where c is the number
    of channels provided and numFeatures() is the number of features that are calculated for
    each channel. N is the number of voxels that all belong to the same
    supervoxel.i

    The values for each channel are normalized to 0,...,1

    """

    def __init__(self):
        pass

    def name(self):
        return self.__class__.__name__

    def numFeatures(self):
        """
        returns the number of features per channel
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
        return 1


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
        self.binEdges = np.arange(0, 1.0, 1.0/bins)
        self.binEdges = np.append(self.binEdges, [1.0])



    def numFeatures(self):
        return  self.bins

    def features(self, voxels):
        
        nchannels = voxels.shape[1]
        nvoxels = voxels.shape[0]
        nbins = self.bins

        nfeatures = nchannels*self.bins

        features = np.zeros((nfeatures,))

        for i in range(nvoxels):
            features[i*nbins:(i+1)*nbins] = (1.0/nvoxels)*np.histogram(voxels[i], bins=self.binEdges)[0]

        
        return features        


# test the features
if __name__ == "__main__":

    #Test data (n, c), 2 voxels and 3 channels per voxel
    test = np.array([[0.1, 0.2, 0.9], [0.1, 0.2, 0.9]])
    
    mean = MeanChannelValueFeature()
    meanTest = mean.features(test)
    assert(meanTest.shape[0] == 3)


    #Test ChannelHistogramFeature
    hist = ChannelHistogramFeature(20)
    histTest = hist.features(test)
    print histTest
    assert(histTest.shape[0] == 20*3)
    


    
