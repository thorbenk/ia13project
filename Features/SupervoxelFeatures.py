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

    def __init__(self, noAxes=3, calcEigVectors=False):
    
        if noAxes > 3:
            print 'WARNING: max number of principal axes can be 3. Will set axes = 3'
            self.axes = 3
        else:
            self.axes = noAxes
        
        self.calcEigVectors = calcEigVectors

    
    def numFeatures(self):
        if self.calcEigVectors:
            return self.axes + 3*self.axes
        else:
            return self.axes

    def features(self, supervoxel):

        # calculate center of mass (cm)
        cm_x = np.mean(supervoxel[:,0])
        cm_y = np.mean(supervoxel[:,1])
        cm_z = np.mean(supervoxel[:,2])
        cm = np.array([cm_x, cm_y, cm_z])
        # change origin to center of mass
        supervoxel -= cm

        # write coordiates into columns of a matrix
        supervoxel = supervoxel.swapaxes(0, 1)
        # calculate singular value decomposition of supervoxel
        u, s, v = svd(supervoxel, full_matrices=True, compute_uv=True)

        if self.calcEigVectors:
            return np.array(np.concatenate((s[0:noAxes], u[:,0], u[:,1], u[:,2])))
        else:
            return np.array(s[0:noAxes])

## TODO: Convex Hull Volume, Shape Probability, ...

#test the channel generators here
if __name__ == "__main__":

    # test data set
    supervoxel = np.array([[0, 0, 1], [0, 0, 5], [0, 0, -8]])
    # calculate principal components or not
    calcEigVectors = True
    # number of principal component axes to be computed
    noAxes = 1

    ### test size
    print '>>> size test'
    testSize = SizeFeature()
    numFeatures = testSize.numFeatures()
    size = testSize.features(supervoxel)
    assert type(size) == np.ndarray, 'type of size is {0}, should be np.ndarray'.format(type(size))
    print 'numFeatures:', numFeatures
    print 'size:', size
    ###

    ### test PCA
    print '>>> PCA test'
    testPCA = PCA(noAxes=noAxes, calcEigVectors=calcEigVectors)
    numFeatures = testPCA.numFeatures()
    result = testPCA.features(supervoxel)
    assert type(result) == np.ndarray, 'type of result is {0}, should be np.ndarray'.format(type(result))
    print 'numFeatures:', numFeatures
    if calcEigVectors:
        print 'singular values:', result[0:noAxes]
        print 'principal directions:'
        eigVectors = []
        for i in range(noAxes, noAxes+3*noAxes, 3):
            eigVectors.append(result[i:i+3])
            print result[i:i+3]

        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure("Test PCA")
        ax = fig.gca(projection='3d')
        x = supervoxel[:,0]
        y = supervoxel[:,1]
        z = supervoxel[:,2]

        ax.plot(x, y, z, 'ko')
        # plot principal directions
        # iterator for labeling
        i = 0
        # longer lines for better visualization
        stretch = 10
        for vector in eigVectors:
            xcoords = stretch*np.array([-vector[0], vector[0]])
            ycoords = stretch*np.array([-vector[1], vector[1]])
            zcoords = stretch*np.array([-vector[2], vector[2]])
            ax.plot(xcoords, ycoords, zcoords, label='principal direction {0}'.format(i))
            i += 1
        plt.legend()
        plt.show()

    else:
        print 'singular values:', result
    ###
