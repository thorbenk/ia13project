import numpy
import numpy as np
import scipy
import vigra

# Base Classes For Channel Generators
class IntrinsicChannelGenerator:
    """
    Intrinsic Channel Generators create channel information from the given  3D raw
    data.
    """

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



# Concrete Implementations
# We maybe want to split this up

##
### External Channel Generators
##

class H5ReaderGenerator(ExternalChannelGenerator):
    """
    Reads channels from a HDF file
    This can be used, for instance, to load ilastik-classified labels for the
    type of organ.
    """
    def __init__(self, path):
        """
        path: the path to the HDF-file
        """

        #TODO implement reading of HDF-file
        self.path = path
        self.h5file = None
        self.numchannels = 1

    def channels(self):
        #TODO implement this function
        raise NotImplementedError()

    def numChannels(self):
        return self.numchannels


##
### Intrinsic Channels generators, that create channels from the raw data
##

class TestChannelGenerator(IntrinsicChannelGenerator):

    def __init__(self, numChannels = 3):
        self.numChans = numChannels

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

################################################################################
# TODO: Gradient Amplitudes, Eigenvalues of Hessian of Gaussian, ...?
# I think we want to use vigra filters for this.
# 
# However, we should consider possible speedups. So, when implementing your
# classes, it takes an scale-attribute. Please make sure, that is has a default
# value. 
# 
# If the default value is used, then the filter-kernel size of the particular
# filter should be minimal, so for 3d-input image it should be 3x3x3 for a
# laplacian filter.
# Then we can use scalers later on to speed things up, as we don't have to
# compute the convolution over a wide window.


class LaplaceChannelGenerator(IntrinsicChannelGenerator):
    """
    Computes the laplacian of a given 3D-Image.
    This channel generator can work on different scales by setting a
    gaussian-window width sigma at the constructor.
    """

    def __init__(self, scale = 1.0):
        self.scale = scale

    def channels(self, image):

        #TODO implement me
        # Karsten: this does not work for me here. I don't know how to invoke
        # this function correctly. However, the idea should be clear.

        # laplace = vigra.filters.laplacianOfGaussian(image, scale=self.scale)
        # return np.reshape(laplace, image.shape+(1,))


        # look at the abstract class definition for the shape to return
        return np.reshape(image, image.shape+(1,))

    def numChannels(self):
        return 1


# test the channels generators here
if __name__ == "__main__":

    #Test data (n, n, n), random sampled data
    test = np.random.random((10,10,10))

    print "Test data", test.shape, type(test)
    #One might also wants to add meaningful test data to test the gradient etc...


    laplace = LaplaceChannelGenerator(5.0)
    laplaceChannels = laplace.channels(test)
    # check that output has right shape
    assert( laplace.numChannels() == 1)
    assert( laplaceChannels.shape == (test.shape + (laplace.numChannels(),) ))


    testGenerator = TestChannelGenerator(4)
    testChannels = testGenerator.channels(test)
    assert( testGenerator.numChannels() == 4)
    assert( testChannels.shape == (test.shape + (testGenerator.numChannels(),) ))
    


