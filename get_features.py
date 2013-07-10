import numpy
import h5py
import sys
import vigra
from scipy import stats

import Features

###############################
## Feature Calculation
###############################
##
## This code mostly stolen from get_superpixels.py
## See down below to see how to invoke Feature detection



# path to raw data file
d_path = "data/block00.h5"
d_h5_path = "volume/data"

# path to store features in
f_path = "data/features00.h5"

###############################
###############################


print "Starting to calculate segmentation..."

#Parameters
#Set  plotall to 1 to show superpixels
plotall = 1

#Subsampling
nx = 255
ny = 255
nz = 60

#Import data    
f = h5py.File(d_path, 'r')

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
    

d = f[d_h5_path].value

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


#Find local minima
locmin = vigra.analysis.extendedLocalMinima3D(dg_s.astype(numpy.float32))
dg_s = vigra.analysis.labelVolumeWithBackground(locmin.astype(numpy.float32))



#Watershed
ws, maxRegionLabel = vigra.analysis.watersheds(dg.astype(numpy.uint8),neighborhood = 6, seeds=dg_s,method = 'Turbo')




#####################################################
# Features 
#####################################################

print "Starting to Calculate Features on Supervoxels..."


proc = Features.Processing()

# currently we don't need scalers.
#proc.addScaler(Features.GaussianScaler(2.0))

proc.addChannelFeature(Features.MeanChannelValueFeature())

proc.addChannelGenerator(Features.TestChannelGenerator())

# Adds some channel generators
for scale in [1.0, 5.0]:
    proc.addChannelGenerator(Features.LaplaceChannelGenerator(scale))
    proc.addChannelGenerator(Features.GaussianGradientMagnitudeChannelGenerator(scale))
    proc.addChannelGenerator(Features.EVofGaussianHessianChannelGenerator(scale))

proc.addSupervoxelFeature(Features.SizeFeature())
proc.addSupervoxelFeature(Features.PCA())

features = proc.process(d, ws)


g = h5py.File(f_path, 'w')
g.create_dataset("features", data=features)
g.close()

print "Feature Array Shape: ", features.shape
