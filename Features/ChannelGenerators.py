import numpy
import numpy as np
import scipy
import h5py
import vigra
from vigra.filters import *

# Base Classes For Channel Generators
class IntrinsicChannelGenerator:
    """
    Intrinsic Channel Generators create channel information from the given  3D raw
    data.
    """
    def name(self):
        """ 
        returns the name of the channel to use i plots etc
        """
        raise NotImplementedError() 

    def numChannels(self):
        """
        returns the number of channels that are going to be calculated from
        input image.

        For for gradient magnitude this will be 1, for hessian eigenvalues this
        will be 3, etc...
        """
        raise NotImplementedError() 
    
    def channels(self, image):
        """
        image: 3-dimensional image. only gray-scaled => shape = (x,y,z).

        If you reimplement this function, make sure that the returns shape of
        numpy array is (x, y, z) + (numChannels(),) where numChannels() is the number of channels you
        are generating from the raw data.
        """
        raise NotImplementedError()


class ExternalChannelGenerator:

    def numChannels(self):
        """
        Returns the number of channels that this channel generator will inject
        """
        raise NotImplementedError()

    def channels(self):
        """
        If you reimplement this function, make sure that the returns shape of
        numpy array is (x, y, z) + (numChannels(),) where numChannels() is the number of channels you
        are generating from the raw data.
        """        
        raise NotImplementedError()


##
### External Channel Generators
##

class H5ReaderGenerator(ExternalChannelGenerator):
    """
    Reads channels from a HDF file
    This can be used, for instance, to load ilastik-classified labels for the
    type of organ.
    """
    def __init__(self, path, data='/'):
        """
        path: the path to the HDF-file
        data: the internal HDF path to the dataset
        """

        self.path = path
        self.data = data
        
        # read in data
        f = h5py.File(path, 'r')
        self.data = f[data].value

        # if the data is only 3-dimensional, we need to artificially add a
        # fourth dimension, to comply to the definition of
        # ExternalChannelGenerator

        if(len(self.data.shape)):
            self.data = np.reshape(self.data, self.data.shape+(1,))
        
        # the number of the fourth dimension tells us about the number of
        # provided channels
        self.numchannels = self.data.shape[3]

    def channels(self):
        return self.data

    def numChannels(self):
        return self.numchannels


##
### Intrinsic Channels generators, that create channels from the raw data
##

class TestChannelGenerator(IntrinsicChannelGenerator):

    def __init__(self, numChannels = 3):
        self.numChans = numChannels
        self.scale = 1.0
        
    def name(self):
        return "Test Channel" 
        
    """
    Test Channel Generator
    Note: only for testing purpose.
    It generates two channels that are a multiple of the original data
    """
    def channels(self, image):
        array = np.zeros(image.shape+(self.numChans,))
        for i in range(0, self.numChans):
            array[:,:,:,i] = (i+1)*image
        return array

    def numChannels(self):
        return self.numChans
        

class SmoothImageChannelGenerator(IntrinsicChannelGenerator):
    """
    Computes the smoothed 3D-Image.
    """

    def __init__(self, scale = 1.0):
        self.scale = scale
        
    def name(self):
        return "Smoothed Image" 

    def channels(self, image):
        smooth = gaussianSmoothing(image,self.scale)
        return np.reshape(smooth, image.shape+(1,))
        
    def numChannels(self):
        return 1
        

class LaplaceChannelGenerator(IntrinsicChannelGenerator):
    """
    Computes the laplacian of a given 3D-Image.
    This channel generator can work on different scales by setting a
    gaussian-window width sigma at the constructor.
    """

    def __init__(self, scale = 1.0):
        self.scale = scale
        
    def name(self):
        return "Laplacian" 
        
    def channels(self, image):
        laplace = laplacianOfGaussian(image,self.scale)
        return np.reshape(laplace, image.shape+(1,))

    def numChannels(self):
        return 1
        

class GaussianGradientMagnitudeChannelGenerator(IntrinsicChannelGenerator):

    """
    Computes Gaussian gradient magnitudes for a given 3D-Image.
    """

    def __init__(self, scale = 1.0):
        self.scale = scale
        
    def name(self):
        return "Gaussian Gradient Magnitude" 
        
    def channels(self, image):
        magnitudes = gaussianGradientMagnitude(image, self.scale)
        return magnitudes.reshape(image.shape+(1,))

    def numChannels(self):
        return 1


class EVofGaussianHessianChannelGenerator(IntrinsicChannelGenerator):
    """
    Computes Eigen values of the Hessian of Gaussian matrix for a given 3D-Image.
    """

    def __init__(self, scale = 1.0):
        self.scale = scale
   
    def name(self):
        return "Eigen Values of the Hessian of Gaussian" 
    
    def channels(self, image):
        hessianEVs = hessianOfGaussianEigenvalues(image, self.scale)
        return hessianEVs

    def numChannels(self):
        return 3


class EVofStructureTensorChannelGenerator(IntrinsicChannelGenerator):
    """
    Computes Eigen values of the structure tensor for a given 3D-Image.
    """

    def __init__(self, scale = 1.0):
        self.scale = scale
        
    def name(self):
        return "Eigen Values of the Structure Tensor" 
   
    def channels(self, image):
        structureEVs = structureTensorEigenvalues(image, self.scale, self.scale) 
        return structureEVs

    def numChannels(self):
        return 3


# test the channels generators here
if __name__ == "__main__":
    
    #Test data (n, n, n), random sampled data
    n = 20
    scale = 1.0 # n=5 -> sc<1.1;  n=20 -> sc<6.1  ??? (kernel longer than line) 
    test = np.float32(np.random.random((n,n,n)))
    
    print "Test data", test.shape, type(test), type(test[0,0,0])
    print "Scale: ", scale
    #One might also wants to add meaningful test data to test the gradient etc...
    
    testGenerator = TestChannelGenerator(4)
    testChannels = testGenerator.channels(test)
    # check that output has right shape
    assert( testGenerator.numChannels() == 4)
    assert( testChannels.shape == (test.shape + (testGenerator.numChannels(),) ))
    
    smooth = SmoothImageChannelGenerator(scale)
    smoothChannels = smooth.channels(test)
    assert( smooth.numChannels() == 1)
    assert( smoothChannels.shape == (test.shape + (smooth.numChannels(),) ))
    
    laplace = LaplaceChannelGenerator(scale)
    laplaceChannels = laplace.channels(test)
    assert( laplace.numChannels() == 1)
    assert( laplaceChannels.shape == (test.shape + (laplace.numChannels(),) ))
    
    gauss = GaussianGradientMagnitudeChannelGenerator(scale)
    gaussChannels = gauss.channels(test)
    assert( gauss.numChannels() == 1)
    assert( gaussChannels.shape == (test.shape + (gauss.numChannels(),) ))
    
    hessian = EVofGaussianHessianChannelGenerator(scale)
    hessianChannels = hessian.channels(test)
    assert( hessian.numChannels() == 3)
    assert( hessianChannels.shape == (test.shape + (hessian.numChannels(),) ))
    
    struct = EVofStructureTensorChannelGenerator(scale)
    structChannels = struct.channels(test)
    assert( struct.numChannels() == 3)
    assert( structChannels.shape == (test.shape + (struct.numChannels(),) ))

    

    ## H5ReaderGenerator
    # 
    # NOTE: test worked, it is deactivated however, as the file is not present
    #       in default repository

    #h5Reader = H5ReaderGenerator("../data/block00.h5", "volume/data")
    #h5ReaderChannels = h5Reader.channels()
    #assert(len(h5ReaderChannels.shape) == 4)
    #assert(h5Reader.numChannels() == h5ReaderChannels.shape[3])


    
    print "All tests passed"


