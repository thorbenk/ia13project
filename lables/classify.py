import vigra
import numpy as np
import h5py

#from lables.h5combine import slices
slices = [0,50,100,130,200]

# Returns classifier trained on given data
def learn(trainingData, labels):  #, labelFn, labelData
    RandomForest = vigra.learning.RandomForest()
    RandomForest.learnRF(trainingData, labels)
    return RandomForest

def classify(classifier, testData):
    return classifier.predictProbabilities(testData)

def learn_func(category_vector):
    
    featureFn = "../data/featuresk0.h5"
    featureDataPath = "features"

    f = h5py.File(featureFn, 'r')
    allfeatures = f[featureDataPath].value
    
    nvoxels = allfeatures.shape[0]
    nfeatures = allfeatures.shape[1]
    
    nLabeledVoxels = np.sum(category_vector != 0)

    features = np.zeros((nLabeledVoxels,nfeatures),dtype=np.float32)
    labels = np.zeros((nLabeledVoxels,1),dtype=np.uint32)

    j=0
    for i in range(nvoxels):
        if i < category_vector.shape[0]:
            
            category = category_vector[i]
            
            if(category != 0):
                features[j,:] = allfeatures[i,:]
                labels[j] = category_vector[i]
                j = j+1
    
    
    print "There are",j,"training examples"

    print "Distribution of labels", np.bincount(np.int32(labels.flat))

    RF = learn(features, labels)


    #probabilities = classify(RF, np.float32(allfeatures))
    #print probabilities
    
    return RF

