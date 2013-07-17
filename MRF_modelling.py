from get_graphical_model import getGraphicalModel
from _adjGraph import adjGraph
import h5py
import numpy as np
import cPickle as pickle
import opengm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def testAdjGraph():
  f = h5py.File("data/block00.h5", 'r') # load image data
  d = f["volume/data"].value
  ws = h5py.File("data/ws.h5", 'r') # load watershed segmentation / superpixels
  sp = ws["ws"].value
  
  edges, surfaces = adjGraph(sp, d.astype(np.float32), True) # compute region adjecency graph
  
  g = h5py.File("data/edges.h5", 'w')
  g.create_dataset("edges", data=edges)
  g.close()
  # e = h5py.File("data/edges.h5", 'r')
  # edges = e["edges"].value
  
  pickle.dump(surfaces, open("data/surfaces.p", "wb"))
  #surfaces = pickle.load(open("data/surfaces.p", "rb"))


def printSurf(surf):
  fig = plt.figure("EdgePoints", figsize=(7,7)) # create a figure
  ax=fig.add_subplot(111, projection="3d") # create subplot 3d in figure
  ax.scatter(xs=surf[:,0], ys=surf[:,1], zs=surf[:,2])
  plt.draw()

def getData():
  l = h5py.File("data/labels.h5", "r")
  labels = l["labels"].value
  labels = labels/labels.sum(axis=1)
  
  f = h5py.File('data/block00.h5', 'r')
  data = f['volume/data'].value
  
  d = h5py.File("data/ws.h5",'r')
  sp = d['ws'].value
  edges = adjGraph(sp, data, False)

  return data, sp, edges, labels

def MRF_modelling(superpixels, labels, edges, data, p=0.2):
 # p      : smoothing parameter
 # Labels : is supposed to be an array: length = #SP, width= #Labels with the probaility for each label and each SP  
  labels = l["labels"].value
  import pdb
  pdb.set_trace()

  labels = labels/labels.sum(axis=1)

  nVariables  = superpixels.max()  
  sameLabel  = np.zeros((nVariables,1))
  diffLabel  = np.ones((nVariables,1)) * p
  edgeValues = np.concatenate((sameLabel, diffLabel), axis=1)
     
      
  gm = getGraphicalModel(
    nLabels     = labels.shape[1],   # The width of the labels array is the number of different labels
    nVariables  = superpixels.max(), # The max number of the sp ID is the total number of SPs
    nEdges      = edges.shape[0],    # Number of edges = lenght of edges-array
    edgeVis     = edges[:,0:2],      # Vertice-Tuples in Edges list
    unaryValues = labels,
    edgeValues  = edgeValues,
    gmOperator  = 'adder')
  
  infer = opengm.inference.AlphaBetaSwap(gm)
  infer.infer()
  result = infer.arg()

   
  return result


data, sp, edges, labels = getData()
MRF_modelling(sp, labels, edges, data)
