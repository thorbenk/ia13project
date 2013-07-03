


class AbstractScaler:
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


def SubsamplingScaler(AbstractScaler):
    """
    Scales the image by subsampling it by the given steps
    Note: returned image is smaller than given.
    """
    def __init__(self, step = 2):
        self.step = step

    def scaled(self, image):
        return AbstractScaler.scaled(self, image)
        
