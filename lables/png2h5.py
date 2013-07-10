from os import listdir, getcwd
from os.path import isfile, join

import numpy as np
import vigra
import h5py
import time
import os

"""
neeed specific names!

block00_<slice>_labels.png


0 
1 Cell interior   white    255 255 255
2 Cell wall       blue       0   0 255
3 Mitocondria     red      255   0   0
4 Vesciles        yellow   255 255   0
5 PSD             green      0 255   0
"""

labels = np.array(
      
    ( 
      (255, 255, 255,0),
      (0  ,   0, 255,0),
      (255,   0,   0,0),
      (255, 255,   0,0),
      (  0, 255,   0,0)))

four_zeros = np.zeros(4)


def getLabel(rgb):
    print rgb
    res = labels  - rgb
    res = np.apply_along_axis(np.linalg.norm,1,res)

    return np.argmin( res ) +1
	

def filename2addr(filename):
	s = filename.split("_")
	return s[:2]

def file2array(path):
    img = vigra.impex.readImage(path).view(np.ndarray)
    #img = np.zeros((255,255,4))
    label_img = np.zeros(img.shape[:2])
    #print label_img.shape
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            cur_rgb = img[x][y]
            if (cur_rgb != four_zeros).all():
                label_img[x,y] = getLabel(img[x][y])
            else:
                label_img[x,y] = 0;

    return label_img
	
if __name__ == "__main__":
    mypath = getcwd()
    files = [ f for f in listdir(mypath) if ( isfile(join(mypath,f)) and f.endswith("labels.png")) ]
    
    name = "block00"
    filename = os.path.join("../","data",name+".h5")
    filename_out = os.path.join("../","data",name+"_labels.h5")

    f = h5py.File("../data/block00.h5",'r')
    data_shape = f['volume']['data'].shape
    #data_shape = (255,255,255)

    label_ar = np.zeros(data_shape)
    #print label_ar.shape

    for f in files:
        print f
        h5file,slice = filename2addr(f)
        res =  file2array(f)
        
        #import pdb
        #pdb.set_trace()
        
        label_ar[:,:,slice] =  res 
    f2 = h5py.File(filename_out, 'w')
    f2.create_dataset("volume/data", data=label_ar.astype(np.uint8))
    f2.close()

