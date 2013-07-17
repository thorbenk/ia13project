import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plot
import numpy
import h5py
import sys
import vigra
from scipy import stats

def computeSupervoxels(blockName, slicing):
    z = 42
    
    #Parameters
    #Set  plotall to 1 to show superpixels
    plotall = 0
    
    #Subsampling
    
    #Import data    
    f = h5py.File('data/'+blockName+'.h5', 'r')
    d = f["volume/data"].value
    print "loaded raw data with shape=%r" % (d.shape,)
    f.close()

    slicing = (slice(0,200), slice(0,200), slice(0,100)) #FIXME
    print "apply slicing =", slicing
    d = d[slicing]

    vigra.impex.writeImage(d[:,:,z], "sv_00.jpg", compression='90')
    
    nx, ny, nz = d.shape
    
    #Hesse-Matrix (Edge Detection)
    sig1 = 0.5
    sig2 = 6
    
    #Smoothing
    sigma = 0.1  
    
    #Find Seeds for watershed
    #Hesse-Matrix for seed detection
    sig1b = 2
    sig2b = 9
    
    #Smoothing for seed detection
    sigmab = 0.1
        
    """List of Figures:
        Figure 1: Pretreated Data (enhanced edges)
        Figure 2: Pretreated Data for Seed detection (less edges)
        Figure 3: Detected Seeds for watershed
        Figure 4: 
        Figure 5:
            
    """
    dg = numpy.zeros((nx,ny,nz))
    dg_s = numpy.zeros((nx,ny,nz))    
    dg_th = numpy.zeros((nx,ny,nz))
    
    #Hesse-Matrix
    print "hessian of gaussian"
    dg = vigra.filters.hessianOfGaussianEigenvalues(d.astype(numpy.float32),scale=sig2)[:,:,:,0]

    fname = "sv_01.jpg" 
    print "writing image '%s'" % (fname,)
    vigra.impex.writeImage(dg[:,:,z], "sv_01.jpg", compression='90')

    print "dg.min()= %f, dg.max()=%f" % (dg.min(), dg.max())
    
    #Rescaling    
    dg = dg/numpy.max(dg)*255

    #Thresholding
    #threshold = 30
    #dg_th = stats.threshold(dg,threshmin=0,threshmax=threshold,newval=255)    
    #dg_th = stats.threshold(dg_th,threshmin=threshold,threshmax=255,newval=0)
    #dg = dg_th
        
    
    #Smoothing
    dg = vigra.filters.gaussianSmoothing(dg.astype(numpy.float32),sigma)    

    #Invert data
    dg = numpy.max(dg)-dg
    
    
    #Plot pretreated Data
    if plotall==1:
        plot.figure()
        plot.gray()
        plot.imshow(dg[:,:,nz/2])
    
    temp = dg[:,:,nz/2]
    
    ################################
    #Find Seeds
    #Hesse-Matrix
   
    dg_s = vigra.filters.gaussianSmoothing(d.astype(numpy.float32),sig1b)    
    for i in range (0,d.shape[2]):    
        dg_s[:,:,i] = numpy.abs(numpy.sum(vigra.filters.hessianOfGaussian(d[:,:,i].astype(numpy.float32),sigma = sig2b),axis=2))    
    
    #Rescaling    
    dg_s = dg_s/numpy.max(dg_s)*255

    #Smoothing
    dg_s = vigra.filters.gaussianSmoothing(dg_s.astype(numpy.float32),sigmab)    
    
    #Invert data
    dg_s = numpy.max(dg_s)-dg_s
    
    #Plot central slice 
    if plotall==1:    
        plot.figure()
        plot.gray()
        plot.imshow(dg_s[:,:,nz/2])

    #Find local minima
    locmin = vigra.analysis.extendedLocalMinima3D(dg_s.astype(numpy.float32))
    dg_s = vigra.analysis.labelVolumeWithBackground(locmin.astype(numpy.float32))
    if plotall==1:    
        plot.figure()
        plot.gray()
        plot.imshow(dg_s[:,:,nz/2])
    
    
    ################################
    
    #Watershed
    ws, maxRegionLabel = vigra.analysis.watersheds(dg.astype(numpy.uint8),neighborhood = 6, seeds=dg_s,method = 'Turbo')
    
    
    #Export Data
    g = h5py.File("data/ws.h5", 'w')
    g.create_dataset("ws", data=ws)
    g.close()
    
    #Plot central slice 
    if plotall==1:    
        plot.figure()
        plot.gray()
        plot.imshow(d[:,:,nz/2])


    #Plot superpixels
    if plotall==1:    
        plot.figure()
    ctable = numpy.random.random((ws.max()+1, 3))
    print ws.shape
    print ctable.shape
    c = ctable[ws]
    
    
    for i in range(0,3):
        c[:,:,:,i] = c[:,:,:,i]*d/255
    c = c/numpy.max(c)   
        
    print c.shape
    
    if plotall==1:
        plot.imshow(c[:,:,30])
        plot.show()

    print ws.shape

if __name__ == '__main__':
    from get_config import small_slicing
    computeSupervoxels('block00', small_slicing) 
