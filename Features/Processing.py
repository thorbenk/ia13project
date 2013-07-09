import numpy as np
import numpy

import Scalers
import ChannelGenerators
import ChannelFeatures
import SupervoxelFeatures


class Processing:
    """

    The basic idea here is:
        - Create channels from Input Image like Laplacian, Gradient
        Magnitude, ..., loaded from harddisk.
        They are all voxel-wise. This is done by ChannelGenerators.

        - Scale input data beforhand: there are several possible ways of scaling
        an image. This however only works, if the dimension of the image will
        not be changed. The scaling will be done by Scalers. This has the
        advantage, that channel generator, that don't have an own scaling
        mechanism, can share precomputed scales of the raw data. This safes
        computation time. 

        However, the scaling can also be handy for convolution filters, as you
        can minimize the filter kernel and so the computation time will be
        smaller as well.

        - Calculate some statistics of the channels. Do this Supervoxel-wise.
        This is done by ChannelFeatures.

        - Finally calculate Supervoxel-Features like Size, volume of convex
        hull, PCA, ... This will be done by SupervoxelFeatures. They might also
        use Scalers beforhand. 



    This class is responsible for the whole feature processing step.

    It schedules the processing for each channel generator.
    If the channel generator is an instrinsic channel generator, than it also
    receives scaled input, if desired.

    Afterwards it linearizes the memory for each supervoxel, so that channel
    features might work faster.

    It then calls the channel features (statistic on channels) and concatenates all
    channel features for each supervoxel into one vector.

    Lastly, the supervoxel features will be calculated.

    All features for each supervoxel then will be merged.

    """

    # instances for Scalers, ChannelGenerators, ChannelFeatures and 
    # SuperpixelFeatures...
    scalers = []

    scalingChannelGenerators = []    
    channelGenerators = []

    channelFeatures = []
    supervoxelFeatures = []


    def __init__(self):
     
        
        pass

    def addScaler(self, scaler):
        """
        scaler: the scale you want to add
        """

        assert(isinstance(scaler, Scalers.AbstractScaler))
        self.scalers.append(scaler)

    def addChannelGenerator(self, gen, scaling=False):
        """
        gen: the channel genenerator
        scaling: if True, than the input for the given channel generator
                 will also be scaled according to all initialized scalers
        """

        # make sure we only get instances that are of these kind of classes
        assert(isinstance(gen, ChannelGenerators.IntrinsicChannelGenerator) or
               isinstance(gen, ChannelGenerators.ExternalChannelGenerator))


        if scaling:
            # scaling only works for intrinsic channel generators
            assert(isinstance(gen, ChannelGenerators.IntrinsicChannelGenerator))
            self.scalingChannelGenerators.append(gen)
        else:
            self.channelGenerators.append(gen)


    def addChannelFeature(self, feature):
        """
        feature: the channel feature you want to add
        """
        assert(isinstance(feature, ChannelFeatures.ChannelFeature))
        self.channelFeatures.append(feature)


    def addSupervoxelFeature(self, feature):
        """
        feature: the supervoxel feature you want to add
        """
        self.supervoxelFeatures.append(feature)

    
    def process(self, image, labels):
        """
        this does the processing magic.
        image: 3d-image of cells
        labels: labels for each supervoxel after seperation

        Note: image.shape == labels.shape
        """

        #make sure that we have an image with exactly 3 dimensions,
        # as given by the example data blocks
        assert(len(image.shape) == 3)
        assert(image.shape == labels.shape)
        #assert(labels.type is np.uint32)
        
        #bc = np.bincount(labels.flatten())
        nLabels = labels.max()
        
        
        ### Preparation Phase
        ##  Scale Image according to scalers
        ##  and calculate the channels

        # generate channels
        channels = []

        # NOTE: if we want to allow scaled images in terms of changed
        # dimensions, then you have to redefine your Scalers in order to also
        # return the new labels. Furthermore, we have to store generated
        # channels for each scaled version. This becomes a little messy then. So
        # for now we only allow scalers that don't change dimension.
        
        
        for cg in self.channelGenerators:
            # we need to distinguish here between intrinsic and external channel
            # generators
            if(isinstance(cg, ChannelGenerators.IntrinsicChannelGenerator)):
                curChannel = cg.channels(image)
                channels.append(curChannel)
            
            else:
                # this is an external channel generator
                # first check if given channels match the shape of given
                # intrinsic data. This is needed to be able to match against the
                # labels.
                curChannel = cg.channels()
                if(curChannel.shape[0:3] == image.shape[0:3]):
                    channels.append(curChannel)
                else:
                    raise IndexError()
        
        for scaler in self.scalers:
            scaledImage = scaler.scaled(image)
            for cg in self.scalingChannelGenerators:
                channels.append(cg.channels(image))
        
        # make given channels to be a numpy array
        # do this by joining the last axis. This makes sense, as the last axis
        # contains the channels that were created by each channel generator
        #NOTE: if scaled images with different dimensions should be allowed,
        #      than this would not be possible that easy anymore.

        channels = np.concatenate(channels, axis=3)
        nChannels = channels.shape[3]
        
        #store for all supervoxels the indices of voxels
        supervoxels = []
        for label in range(1, nLabels+1):
            indices = np.where(labels == label)
            supervoxels.append(indices)
        
        
        ### linearize channel data per supervoxel and calculate features
        #  By linearizing the data according to supervoxels we can speed up
        #  things heavily as we don't need to jump around in memory all the
        #  time.
        #
          
        
        cfeatures = []
        sfeatures = []

        # go through all supervoxels
        # NOTE: if dimension-scaled images should be allowed here, the handling of
        #       different dimensions has to be taken into account as well.

        # FIXME: concurrent computation over supervoxels.
        #        we can handle this in python 3.2 quite easily by the help of
        #        concurrent-frame work.

        for label in range(0,nLabels):
            # FIXME: this still does not linearize in memory
            voxel = channels[supervoxels[label]]

            # get the list of points belonging to the given supervoxel
            coordinates = np.transpose(supervoxels[label])
            
            localSFeatures = []
            for sfeature in self.supervoxelFeatures:
                localSFeatures.append(sfeature.features(coordinates))

            # merge all different features per supervoxel into one list
            localSFeatures = np.concatenate(localSFeatures)
            sfeatures.append(localSFeatures)

            #go through all channel features
            localCFeatures = []
            for cf in self.channelFeatures:
                localCFeatures.append(cf.features(voxel))

            #merge all different features per supervoxel into one list
            localCFeatures = np.concatenate(localCFeatures)
            cfeatures.append(localCFeatures)

        # make the list to be numpy arrays
        cfeatures = np.array(cfeatures)
        sfeatures = np.array(sfeatures)
        
        # concat all values into one big thing
        concats = []
        if(len(cfeatures) > 0):
            assert(len(cfeatures.shape) == 2)
            concats.append(cfeatures)
        if(len(sfeatures) > 0):
            assert(len(sfeatures.shape) == 2)
            concats.append(sfeatures)
        
        allFeatures = np.concatenate(concats, axis=1)

        return allFeatures


        

if __name__ == "__main__":

    # create test data.
    n = 100
    image = np.float32(np.random.randint(0, 255, size=(n, n, n)))
    image = np.float32(10*np.ones((n,n,n)))
    
    # create some supervoxels
    labels = np.ones((n, n, n), dtype=np.uint32)
    labels[0:50,   0:50,       0:50] = 2
    labels[50:100, 0:50,       0:50] = 3
    labels[0:50,   50:100,     0:50] = 4
    labels[0:50,   50:100,   50:100] = 5
    labels[50:100,   50:100,   50:100] = 6

    image[np.where(labels==2)] = 20
    image[np.where(labels==3)] = 30
    image[np.where(labels==4)] = 40
    image[np.where(labels==5)] = 50
    image[np.where(labels==5)] = 60

    proc = Processing()

    # currently we don't need scalers.
    #proc.addScaler(Scalers.DummyScaler())

    proc.addChannelFeature(ChannelFeatures.MeanChannelValueFeature())

    # Adds some channel generators
    proc.addChannelGenerator(ChannelGenerators.TestChannelGenerator())
    for scale in [1.0, 5.0, 10.0]:
        proc.addChannelGenerator(ChannelGenerators.LaplaceChannelGenerator(scale))
        proc.addChannelGenerator(ChannelGenerators.GaussianGradientMagnitudeChannelGenerator(scale))
        proc.addChannelGenerator(ChannelGenerators.EVofGaussianHessianChannelGenerator(scale))

    proc.addSupervoxelFeature(SupervoxelFeatures.SizeFeature())
    proc.addSupervoxelFeature(SupervoxelFeatures.PCA())
    ret = proc.process(image, labels)
    print "###################"
    print ret


