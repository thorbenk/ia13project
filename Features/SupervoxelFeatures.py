# Supervoxel Features take a whole supervoxel and try to extract information
# from the shape.

class SupervoxelFeature:
    """
    calculates some features from a whole given supervoxel
    """

    def __init__(self):
        pass

    def name(self):
        """
        Returns the name of feature.
        Can be used for debugging purpose later on.
        """
        return self.__class__.__name__

    def numFeatures(self):
        """
        this function should return the number of features, that are calculated
        per supervoxel
        """
        raise NotImplementedError()

    def features(self, points):
        """
        computes supervoxel features
        parameters:
        -points: an (n, 3) shaped array. n is the number of points.

        returns: an (c,) shaped array where c is the number of features.
        """

        raise NotImplementedError()




#class SizeFeature(SupervoxelFeature):
#    
#    def numFeatures(self):
#        return 1
#
#    def features(self, supervoxel):
#        return array([100])
    

## TODO: PCA, Convex Hull Volume, Shape Probability, ...



#test the channel generators here
if __name__ == "__main__":
    pass


