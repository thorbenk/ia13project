# Scalers take raw data in scale them respectively. This way we can retrieve
# information on different level of details
#
# We want to use dedicated scalers in order to speed things up. So we only need to
# scale the image once and all channel generators can share one scaled version.

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



class DummyScaler(AbstractScaler):
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
        return AbstractScaler.scaled(self, image)

    def __str__(self):
        return "GScaler["+str(self.sigma)+"]"
