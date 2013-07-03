import numpy as np
import numpy

import Scalers
import ChannelGenerators
import ChannelFeatures



class Processing:
    """

    The basic idea here is:
        - create channels from Input Image like Laplacian, Gradient
        Magnitude, ..., loaded from harddisk.
        They are all voxel-wise. This is done by ChannelGenerators.

        - scale input data beforhand: there are several possible ways of scaling
        an image. This however only works, if the dimension of the image will
        not be changed. The scaling will be done by Scalers.

        - calculate some statistics of the channels. Do this Supervoxel-wise.
        This is done by ChannelFeatures.

        - Finally calculate Supervoxel-Features like Size, volume of convex
        hull, PCA, ... This will be done by SupervoxelFeatures. They might also
        use Scalers beforhand. 



    This class is responsible for the whole feature processing step.

    It schedules the processing for each channel generator.
    If the channel generator is an instrinsic channel generator, than it also
    receives scaled input, if desired.

    Afterwards it linearizes memory in terms of supervoxels.

    It then calls the statistic modules/channel features and concatenates all
    channel features for each supervoxel into one vector.

    [Todo] Lastly, the supervoxel features will be calculated.

    [Todo] All features for each supervoxel then will be merged

    """

    
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
        pass

    
    def process(self, image, labels):
        """
        this does the processing magic.
        image: 3d-image of cells
        labels: labels for each supervoxel after seperation
        """

        #make sure that we have an image with exactly 3 dimensions,
        # as given by the example data blocks
        assert(len(image.shape) == 3)
        assert(image.shape == labels.shape)
        #assert(labels.type is np.uint32)
        
        bc = np.bincount(labels.flatten())
        nLabels = labels.max()
        
        
        ### Preparation Phase
        ##  Scale Image according to scalers
        ##  and calculate the channels

        # generate channels
        channels = []

        # NOTE: if we want to allow scaled images in terms of changed
        # dimensions, then you have to redefine your Scalers in order to also
        # return the new labels. Furthermore, we have to store generated
        # channels for each scaled version. This becomes a little messy then...
        
        
        for cg in self.channelGenerators:
            # we need to distinguish here between intrinsic and external channel
            # generators
            if(isinstance(cg, ChannelGenerators.IntrinsicChannelGenerator)):
                curChannel = cg.channels(image)
                channels.append(curChannel)
            
            else:
                # this is an external channel generator
                # first check if given channels match the shape of given
                # intrinsic data
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
        
        
        ### linearize the channels per supervoxel
        ##  We need to to this for memory optimization
        ##  Possibly this also needs some harddisk caching as not all data might
        ##  fit into memory.
        ##  In turn calculate the channel features.
        
        cfeatures = []
        
        # go through all supervoxels
        # NOTE: if dimension-scaled images should be allowed here, the handling of
        # different dimensions has to be taken into account here as well.

        for label in range(0,nLabels):
            voxel = channels[supervoxels[label]]

            # NOTE: this still does not linearizes memory, it only returns a
            #       view.

            #go through all channel features
            for cf in self.channelFeatures:
                cfeatures.append(cf.features(voxel))
        
        cfeatures = np.array(cfeatures)
        
        #TODO implement supervoxel features
        
        #finally check that output is nSupervoxels*nFeatures
        assert(cfeatures.shape == (nLabels, nChannels))
        
        return np.array(cfeatures)


        

if __name__ == "__main__":

    
    # create test data.
    n = 100
    image = np.random.randint(0, 255, size=(n, n, n))
    image = 5*np.ones((n,n,n))
    labels = np.zeros((n, n, n), dtype=np.uint32)
    labels[0:50,   0:50,       0:50] = 1
    labels[50:100, 0:50,       0:50] = 2
    labels[0:50,   50:100,     0:50] = 3
    labels[0:50,   50:100,   50:100] = 4

    image[np.where(labels==1)] = 20
    image[np.where(labels==2)] = 30
    image[np.where(labels==3)] = 40
    image[np.where(labels==4)] = 50

    
    proc = Processing()
    proc.addScaler(Scalers.DummyScaler())
    proc.addChannelFeature(ChannelFeatures.MeanChannelValueFeature())
    proc.addChannelGenerator(ChannelGenerators.TestChannelGenerator())
    proc.addChannelGenerator(ChannelGenerators.LaplaceChannelGenerator(), False)

    print proc.process(image, labels)


