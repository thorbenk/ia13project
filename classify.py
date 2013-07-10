import vigra
import numpy as np
import h5py

# Returns classifier trained on given data
def learn(trainingData, labels):  #, labelFn, labelData
   RandomForest = vigra.learning.RandomForest()
   RandomForest.learnRF(features, labels)
   return RandomForest

def classify(classifier, testData):
   return classifier.predictProbabilities(testData)


if __name__ == "__main__":

   from get_config import small_slicing

   f1 = h5py.File("data/block00_labels.h5",'r')
   f2 = h5py.File("data/ws.h5","r")

   sp     = f2['ws'][small_slicing]
   labels = f1['volume']['data'][small_slicing]

   slices = [0,25,50]

   index = 0
   lable_table = np.zeros((0,5))

   index_dict = {}

   for i in slices:
      print "slice:",i

      sp_current = sp[:,:,i]
      lb_current = labels[:,:,i]

      for u in np.unique(sp_current):
         index_dict[u] = index

   featureFn = "data/features00.h5"
   featureDataPath = "features"

   f = h5py.File(featureFn, 'r')
   allfeatures = f[featureDataPath].value

   nvoxels = allfeatures.shape[0]
   nfeatures = allfeatures.shape[1]
   nLabeledVoxels = len(index_dict)

   features = np.zeros((nLabeledVoxels,nfeatures),dtype=np.float32)
   labels = np.zeros((nLabeledVoxels,1),dtype=np.uint32)

   j=0
   for i in range(nvoxels):
      if i in index_dict:
         features[j,:] = allfeatures[i,:]
         labels[j] = index_dict[i]
         j = j+1

   RF = learn(features, labels)
   probabilities = classify(RF, np.float32(allfeatures))

   print probabilities
