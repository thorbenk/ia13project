import numpy as np
import vigra
import h5py
import time

import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plot

from png2h5 import labels as label_color
four_zeros = np.zeros(4)

from classify import learn_func

slices = [0,50,100,130,200]

if __name__ == "__main__":
    
    f1 = h5py.File("../data/block00_labels.h5",'r')
    f2 = h5py.File("../data/ws.h5","r")

    sp     = f2['ws']
    labels = f1['volume']['data']
    
    slices = [0,50,100,130,200]
    
    index = 0
    lable_table = np.zeros((0,6),dtype=np.uint32)
 
    index_dict = {}
    
    for i in slices:
        print "slice:",i

        sp_current = sp[:,:,i]
        lb_current = labels[:,:,i]
        
        '''
        #tmp = label_color[ lb_current.transpose()]
        #tmp = tmp[:,:,:3]
        tmp = lb_current.transpose() == 2
        plot.imshow(tmp)
        plot.colorbar()
        plot.show()
        '''

        for u in np.unique(sp_current):
            index_dict[u] = index
            index+=1

        lable_table = np.r_[lable_table,np.zeros((index,6))]
        
        for x in range(lb_current.shape[0]):
            for y in range(lb_current.shape[1]):
                lable = lb_current[x,y]
                spixel = sp_current[x,y]
                #if(lable > 1):
                #    print lable

                #if(lable != 0):
                
                lable_table[ index_dict[spixel], lable] += 1
        
        print "Sigma lables:",np.sum(lable_table,axis=0) 
        

        categories = np.zeros((np.max(sp_current)+1,),dtype=np.uint32)

        for spixel_name in np.unique(sp_current):
            weight = lable_table[index_dict[spixel_name]]
            
            if( sum(weight) >1 ):
                category = np.argmax(weight) 
                categories[spixel_name] = category
        
        
        pict = label_color[categories[sp_current]]
        vigra.impex.writeImage(pict,"test_00_"+str(i)+".png")
    
    category_vector = np.zeros((max(index_dict.keys())+1,6))
    for index,val in index_dict.iteritems():
        category_vector[index] = lable_table[val]
    
    def_category = np.argmax(category_vector,axis=1)
    

    forest = learn_func(def_category)
   
    #Now that we learned the forest we can use this to classify
    #unseen supervoxels

    slices_to_classify = [0,25,125,140,210]
    
 
    featureFn = "../data/featuresk0.h5"
    featureDataPath = "features"
    f = h5py.File(featureFn, 'r')
    allfeatures = f[featureDataPath].value
    
    #import pdb
    #pdb.set_trace()
    
    assert allfeatures.shape[0] == np.max(sp) 
    
    for i in slices_to_classify:
        print "slice:",i

        sp_current = sp[:,:,i]
        lb_image = np.zeros(sp_current.shape)
        

        current_spixels = np.unique(sp_current)
        
        label_vector = np.zeros(np.max(sp_current)+1,dtype=np.uint32)

        for j in range(len(sp_current)):
            label_vector[sp_current[j]] = j
        
        #get the features for these supervolxels
        current_features = np.float32( allfeatures[current_spixels] )
        current_categories = forest.predictProbabilities(current_features)
        cur_label = np.argmax(current_categories,axis=1) + 1

        tmp = cur_label[label_vector[sp_current]]
        pict = label_color[tmp]
        vigra.impex.writeImage(pict,"res_00_"+str(i)+".png")
        
        import pdb
        pdb.set_trace()


