# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:13:00 2013

@author: mfk
"""

from get_graphical_model import getGraphicalModel
from _adjGraph import adjGraph

def MRF_modelling(superpixels, labels, unaries, data):
  edges, surfaces = adjGraph(superpixels, data)
  
  nLabels    = labels.shape[0]#################################################
  nVariables = superpixels.max()###############################################
  nEdges     = edges.shape[0]
  edgeVis    = edges[:,0:2]
  
  unaryValues= unaries#########################################################
  edgeValues = binaryClassifier(edges)#########################################
     
      
  gm = getGraphicalModel(
    nLabels     = nLabels,
    nVariables  = nVariables,
    nEdges      = nEdges,
    edgeVis     = edgeVis,
    unaryValues = unaryValues,
    edgeValues  = edgeValues,
    gmOperator  = 'adder')
    
  gm
    
  return gm

