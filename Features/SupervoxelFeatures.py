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

    def features(self, labels, image):
        """calculates the features for each supervoxel"""

        #TODO input format needs to be defined properly
        return NotImplementedError()




#class SizeFeature(SupervoxelFeature):
#    
#    def numFeatures(self):
#        return 1
#
#    def features(self, supervoxel):
#        return 100
    

## TODO: PCA, Convex Hull Volume, Shape Probability, ...



