


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
