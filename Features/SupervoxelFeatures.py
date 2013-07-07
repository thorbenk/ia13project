import numpy as np
from numpy.linalg import svd

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




class SizeFeature(SupervoxelFeature):
    
    def numFeatures(self):
        return 1

    def features(self, supervoxel):
        return np.array([len(supervoxel)])


# TODO: find principal component directions, too
class PCA(SupervoxelFeature):

    def __init__(self, noAxes=2, calcEigVectors=False):
        self.axes = noAxes
        self.calcEigVectors = calcEigVectors

    def numFeatures(self):
        if self.calcEigVectors:
            return self.axes + 3*self.axes
        else:
            return self.axes

    def features(self, supervoxel):
        if self.axes > 3:
            print 'WARNING: max number of principal axes can be 3. Will set noAxes = 3'
            self.axes = len(supervoxel)

        # calculate center of mass (cm)
        cm_x = np.mean(supervoxel[:,0])
        cm_y = np.mean(supervoxel[:,1])
        cm_z = np.mean(supervoxel[:,2])
        cm = np.array([cm_x, cm_y, cm_z])
        # change origin to center of mass
        supervoxel -= cm

        # write coordiates into columns of a matrix
        supervoxel = supervoxel.swapaxes(0, 1)
        u, s, v = svd(supervoxel, full_matrices=True, compute_uv=True)

        if self.calcEigVectors:
            pass
        else:
            return np.array(s)

## TODO: PCA, Convex Hull Volume, Shape Probability, ...

#test the channel generators here
if __name__ == "__main__":

    supervoxel = np.array([[1, 2, 3], [4, 5, 6], [7, 7, 7], [5, 4, 8]])
    
    # test size
    testSize = SizeFeature()
    numFeatures = testSize.numFeatures()
    size = testSize.features(supervoxel)
    assert type(size) == np.ndarray, 'type of size is {0}, should be np.ndarray'.format(type(size))
    print 'number of features:', numFeatures
    print 'size:', size
    ###

    # test PCA
    testPCA = PCA(noAxes=3, calcEigVectors=False)
    numFeatures = testPCA.numFeatures()
    singValues = testPCA.features(supervoxel)
    assert type(size) == np.ndarray, 'type of size is {0}, should be np.ndarray'.format(type(size))
    print 'number of features:', numFeatures
    print 'singular values:', singValues

    """
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure("Test PCA")
    ax = fig.gca(projection='3d')
    x = supervoxel[:,0]
    y = supervoxel[:,1]
    z = supervoxel[:,2]

    ax.plot(x, y, z, 'ko')
    #plt.show()
    """
    ###
