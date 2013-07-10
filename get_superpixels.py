import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plot
import numpy
import h5py
import sys
import vigra
from scipy import stats

if __name__ == "__main__":
    
    #Parameters
    #Set  plotall to 1 to show superpixels
    plotall = 0
    
    #Subsampling
    
    #Import data    
    f = h5py.File("data/block00.h5", 'r')
    
    #Hesse-Matrix (Edge Detection)
    sig1 = 0.5
    sig2 = 6
    stepS = 1
    winS = 0 
    
    #Smoothing
    sigma = 0.1  
    
    #Find Seeds for watershed
    #Hesse-Matrix for seed detection
    sig1b = 2
    sig2b = 9
    stepSb = 1
    winSb = 0 
    
    #Smoothing for seed detection
    sigmab = 0.1
        
    """List of Figures:
        Figure 1: Pretreated Data (enhanced edges)
        Figure 2: Pretreated Data for Seed detection (less edges)
        Figure 3: Detected Seeds for watershed
        Figure 4: 
        Figure 5:
            
    """
        
    
    
    d = f["volume/data"].value
    print d.shape

    nx = d.shape[0]
    ny = d.shape[1]
    nz = d.shape[2]

    #Subsampling
    d = d[0:nx,0:ny,0:nz]
    dg = numpy.zeros((nx,ny,nz))
    dg_s = numpy.zeros((nx,ny,nz))    
    dg_th = numpy.zeros((nx,ny,nz))
    
    #Hesse-Matrix
    dg = vigra.filters.gaussianSmoothing(d.astype(numpy.float32),sig1)    
    for i in range (0,d.shape[2]):    
        dg[:,:,i] = numpy.abs(numpy.sum(vigra.filters.hessianOfGaussian(d[:,:,i].astype(numpy.float32),sigma = sig2,step_size = stepS,window_size = winS),axis=2))    
    
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
        dg_s[:,:,i] = numpy.abs(numpy.sum(vigra.filters.hessianOfGaussian(d[:,:,i].astype(numpy.float32),sigma = sig2b,step_size = stepSb,window_size = winSb),axis=2))    
    
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
