import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plot
import numpy
import h5py
import sys
import vigra
from scipy import stats

def writeSlice(fname, a, z):
    print "writing image '%s'" % (fname,)
    vigra.impex.writeImage(a[:,:,z], fname, compression='90')

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

    writeSlice("sv_00.jpg", d, z) 
    
    nx, ny, nz = d.shape
    
    #Hesse-Matrix (Edge Detection)
    sig1 = 0.5
    sig2 = 3
    
    #Smoothing
    sigma = 0.1  
    
    #Find Seeds for watershed
    #Hesse-Matrix for seed detection
    sig1b = 2
    sig2b = 6
    
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
    dg = vigra.filters.gaussianGradientMagnitude(d.astype(numpy.float32),sig2)
    
    writeSlice("sv_01.jpg", dg, z) 

    print "dg.min()= %f, dg.max()=%f" % (dg.min(), dg.max())
    
    #Rescaling    
    dg = ((dg-dg.min())/(numpy.max(dg)-dg.min())*255).astype(numpy.uint8)
    writeSlice("sv_02.jpg", dg, z) 

    #Thresholding
    #threshold = 30
    #dg_th = stats.threshold(dg,threshmin=0,threshmax=threshold,newval=255)    
    #dg_th = stats.threshold(dg_th,threshmin=threshold,threshmax=255,newval=0)
    #dg = dg_th
        
    ''' 
    #Smoothing
    dg = vigra.filters.gaussianSmoothing(dg.astype(numpy.float32),sigma)    

    #Invert data
    dg = numpy.max(dg)-dg
    '''
    
    #Plot pretreated Data
    if plotall==1:
        plot.figure()
        plot.gray()
        plot.imshow(dg[:,:,nz/2])
    
    ################################
    #Find Seeds
    #Hesse-Matrix
  
    print "hessian of gaussian, smoothed for seeds"
    dg_s = vigra.filters.gaussianGradientMagnitude(d.astype(numpy.float32),sig2b)
    writeSlice("sv_03.jpg", dg_s, z) 
    
    #Rescaling    
    dg_s = ((dg_s-dg_s.min())/(dg_s.max()-dg_s.min())*255).astype(numpy.uint8)
    writeSlice("sv_04.jpg", dg_s, z) 

    '''
    #Smoothing
    dg_s = vigra.filters.gaussianSmoothing(dg_s.astype(numpy.float32),sigmab)    
    
    #Invert data
    dg_s = numpy.max(dg_s)-dg_s
    '''
    
    #Plot central slice 
    if plotall==1:    
        plot.figure()
        plot.gray()
        plot.imshow(dg_s[:,:,nz/2])

    #Find local minima
    locmin = vigra.analysis.extendedLocalMinima3D(dg_s.astype(numpy.float32))
    seeds = vigra.analysis.labelVolumeWithBackground(locmin.astype(numpy.float32))
    if plotall==1:    
        plot.figure()
        plot.gray()
        plot.imshow(dg_s[:,:,nz/2])

    nSeeds = seeds.max()
    relabel = (numpy.random.random((nSeeds+1, 3))*255).astype(numpy.uint8)
    seedImg = relabel[seeds[:,:,z]]
    print seedImg.shape, seedImg.dtype
    vigra.impex.writeImage(seedImg, "sv_05.jpg", compression='90')
    
    
    ################################
    
    #Watershed
    ws, maxRegionLabel = vigra.analysis.watersheds(dg.astype(numpy.uint8),neighborhood = 6, seeds = seeds, method = 'Turbo')
    
    relabel = (numpy.random.random((maxRegionLabel+1, 3))*255).astype(numpy.uint8)
    
    print relabel
    print ws.shape
    print '######'
    print maxRegionLabel
    
    wsImg = relabel[ws[:,:,z]]
    print wsImg.shape, wsImg.dtype
    vigra.impex.writeImage(wsImg, "sv_06.jpg", compression='90')
    
    
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
