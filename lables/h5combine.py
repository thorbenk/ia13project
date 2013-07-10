import numpy as np
import vigra
import h5py
import time

import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plot

from png2h5 import labels as label_color

label_color = np.r_[np.zeros((1,4)),label_color]
label_color[1:,3] = 255

print label_color

four_zeros = np.zeros(4)

if __name__ == "__main__":
    
    f1 = h5py.File("../data/block00_labels.h5",'r')
    f2 = h5py.File("../data/ws.h5","r")

    sp     = f2['ws']
    labels = f1['volume']['data']
    
    slices = [0,50,100]
    
    index = 0
    lable_table = np.zeros((0,5),dtype=np.uint32)
 
    index_dict = {}
    
    for i in slices:
        print "slice:",i

        sp_current = sp[:,:,i]
        lb_current = labels[:,:,i]
        
        for u in np.unique(sp_current):
            index_dict[u] = index
            index+=1

        lable_table = np.r_[lable_table,np.zeros((index,5))]
        
        for x in range(lb_current.shape[0]):
            for y in range(lb_current.shape[1]):
                lable = lb_current[x,y]
                spixel = sp_current[x,y]
                #if(lable > 1):
                #    print lable

                if(lable != 0):
                    lable_table[ index_dict[spixel], lable] += 1
        
        print "Sigma lables:",np.sum(lable_table,axis=0) 
        categories = np.zeros((np.max(sp_current)+1,),dtype=np.uint32)

        for spixel_name in np.unique(sp_current):
            weight = lable_table[index_dict[spixel_name]]
            
            if( sum(weight) >1 ):
                category = np.argmax(weight) + 1
                categories[spixel_name] = category
        

        pict = label_color[categories[sp_current]]
        vigra.impex.writeImage(pict,"test_00_"+str(i)+".png")

    import pdb
    pdb.set_trace()
