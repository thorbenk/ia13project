import numpy as np
from vigra.filters import *

"""
Scalers take raw data in scale them respectively. This way we can retrieve
information on different level of details

We want to use dedicated scalers in order to speed things up. So we only need to
scale the image once and all channel generators can share one scaled version.
"""

class AbstractScaler:
    """ 
    scales a given image.
    make sure that the size of the input is not changed
    """

    def __init__(self):
        pass

    def scaled(self, image):
        """ this function returns the scaled version of a given 3d-image """
        raise NotImplementedError()

    def __str__(self):
        return "AbstractScaler"


class TestScaler(AbstractScaler):
    """ Dummy Scaler, does nothing """
    def scaled(self, image):
        print "DummyScaler: ", image.shape
        return image

    def __str__(self):
        return "DummyScaler"


# TODO: implement gaussian smoother.
class GaussianScaler(AbstractScaler):
    """
    Scales the image by smoothing with a gaussian that uses the given sigma as
    width.
    Note: returned images has the same size
    """
    def __init__(self, sigma = 1.0):
        self.sigma = sigma

    def scaled(self, image):
        return gaussianSmoothing(image, self.sigma)

    def __str__(self):
        return "GScaler["+str(self.sigma)+"]"



if __name__ == "__main__":

    ## create test data
    # size of the voxel
    n = 11

    # center of the impulse
    c = 5
    test = np.float32(np.zeros((n, n, n)))
    test[c, c, c] = 1

    ## TestScaler
    tscaler = TestScaler()
    tscalerTest = tscaler.scaled(test)
    assert(tscalerTest.shape == test.shape)


    ## GaussianScaler
    # uses gaussian filter kernel
    gscaler = GaussianScaler(1.0)
    gscalerTest = gscaler.scaled(test)
    assert(gscalerTest.shape == test.shape)

    print test[c-1:c+2, c-1:c+2, c-1:c+2]
    print gscalerTest[c-1:c+2, c-1:c+2, c-1:c+2]
