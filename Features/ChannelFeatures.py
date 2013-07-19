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
        
    def displayName(self):
        raise NotImplementedError()

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


class MeanChannelFeature(ChannelFeature):
    """
    Computes for each channel the mean value over all voxels    
    """
    def displayName(self):
        return "Mean"
        
    def features(self, voxels):
        return np.mean(voxels, axis=0)

    def numFeatures(self):
        return 1


class MedianChannelFeature(ChannelFeature):
    """
    Computes for each channel the mean value over all voxels    
    """
    def displayName(self):
        return "Median"
    
    def features(self, voxels):
        return np.median(voxels, axis=0)

    def numFeatures(self):
        return 1


class StdDeviationChannelFeature(ChannelFeature):
    """
    Computes for each channel the mean value over all voxels    
    """
    def displayName(self):
        return "Standard Deviation"
        
    def features(self, voxels):
        return np.std(voxels, axis=0)

    def numFeatures(self):
        return 1
        

class VarianceChannelFeature(ChannelFeature):
    """
    Computes for each channel the mean value over all voxels    
    """
    def displayName(self):
        return "Variance"
        
    def features(self, voxels):
        return np.var(voxels, axis=0)

    def numFeatures(self):
        return 1


class ChannelHistogramFeature(ChannelFeature):
    """
    Computes for each channel a normalized histogram with given number of bins
    """

    def __init__(self, bins=10):
        self.bins = bins
        self.binEdges = np.arange(0, 1.0, 1.0/bins)
        self.binEdges = np.append(self.binEdges, [1.0])

    def displayName(self):
        return "Histogram (" + str(self.bins) + " bins)"
        
    def numFeatures(self):
        return  self.bins

    def features(self, voxels):
        
        nchannels = voxels.shape[1]
        nvoxels = voxels.shape[0]
        nbins = self.bins

        nfeatures = nchannels*self.bins

        features = np.zeros((nfeatures,))

        for i in range(nchannels):
            features[i*nbins:(i+1)*nbins] = (1.0/nvoxels)*np.histogram(voxels[:,i], bins=self.binEdges)[0]

        
        return features        


# test the features
if __name__ == "__main__":

    #Test data (n, c), supervoxel with 2 voxels and 3 channels per voxel
    #test = np.array([[0.1, 0.2, 0.9], [0.1, 0.2, 0.8]])
    test = np.arange(0.0, 0.9, 0.05).reshape((6,3))
    test[2,2] = 1.0
    print test
    
    mean = MeanChannelFeature()
    meanTest = mean.features(test)
    print "mean:"
    print meanTest
    assert(meanTest.shape[0] == 3)
    
    median = MedianChannelFeature()
    medianTest = mean.features(test)
    print "median:"
    print medianTest
    assert(medianTest.shape[0] == 3)
    
    std = StdDeviationChannelFeature()
    stdTest = std.features(test)
    print "std deviation:"
    print stdTest
    assert(stdTest.shape[0] == 3)
    
    var = VarianceChannelFeature()
    varTest = var.features(test)
    print "var:"
    print varTest
    assert(varTest.shape[0] == 3)


    #Test ChannelHistogramFeature
    hist = ChannelHistogramFeature(10)
    histTest = hist.features(test)
    assert(histTest.shape[0] == 10*3)
    print "histogram:"
    for i in range(0,histTest.shape[0], 10):
        print histTest[i:i+10] 
    


    
