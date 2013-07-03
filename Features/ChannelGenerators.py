import numpy
import numpy as np
import scipy


# Base Classes For Channel Generators
class IntrinsicChannelGenerator:

    def __init__(self):
        pass

    def numChannels(self):
        return NotImplementedError() 
    
    def channels(self, image):
        """
        If you reimplement this function, make sure that the return shape of
        numpy array is (x, y, z) + (c,) where c is the number of channels you
        are generating from the raw data.
        """

        return NotImplementedError()


class ExternalChannelGenerator:
    def __init__(self):
        pass

    def numChannels(self):
        return NotImplementedError()

    def channels(self):
        return NotImplementedError()



# Concrete Implementations
# We maybe want to split this up

##
### External Channel Generators
##

class H5ReaderGenerator(ExternalChannelGenerator):
    """
    Reads channels from a HDF file
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

class LaplaceChannelGenerator(IntrinsicChannelGenerator):

    def channels(self, image):
        #TODO: implement laplacian over 3d-image
        print "LaplaceChannelGenerator", image.shape
        return np.reshape(image, image.shape+(1,))

    def numChannels(self):
        return 1


## TODO: Gradient Amplitudes, ...

class TestChannelGenerator(IntrinsicChannelGenerator):
    """
    Test Channel Generator
    Note: only for testing purpose.
    It generates two channels that are a multiple of the original data
    """
    def channels(self, image):
        array = np.zeros(image.shape+(3,))
        array[:,:,:,0] = image
        array[:,:,:,1] = 2.0*image
        array[:,:,:,2] = 3.0*image

        return array

    def numChannels(self):
        return 3

