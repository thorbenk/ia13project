import numpy as np
import vigra
import h5py
import time

import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plot

if __name__ == "__main__":
    
    f = h5py.File("../data/block00.h5",'r')
    
    data = f['volume']['data']
    
    for i in range(0,data.shape[2],10):
        vigra.impex.writeImage(data[:,:,i],"block00_"+str(i)+".png")

        plot.imshow(data[:,:,i])
        plot.show()
        time.sleep(.2)
        plot.clf()
        plot.close()


