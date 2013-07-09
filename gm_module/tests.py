# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:00:29 2013

@author: mfk
"""

import numpy as np
import _adjGraph as gm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

test1 = np.zeros((3,3,3),dtype=np.uint32)
test1[:,:,0] = 1
result = gm.adjGraph(test1, test1.astype(np.float32), False)

points = result[1][0]
fig = plt.figure("EdgePoints", figsize=(10,10)) # create a figure
ax = fig.add_subplot(111, projection="3d") # create subplot 3d in figure
ax.scatter(points[:,0], points[:,1], points[:,2])
print result


# Test functions for the adjGraph-search algorithm #
test1 = np.zeros((3,3,3),dtype=np.uint32)
result = gm.adjGraph(test1, test1.astype(np.float32), True) # should return empty edges and #edges=0
print result

test1[:,:,0] = 1
result = gm.adjGraph(test1, test1.astype(np.float32), True)  # should return edge (0,1) and #edges=1
print result

test1[:,:,2] = 2
result = gm.adjGraph(test1, test1.astype(np.float32), True)  # should return edge (0,1),(0,2) and #edges=2
print result

test1[:,1,:] = 3
result = gm.adjGraph(test1, test1.astype(np.float32), True)  # should return edge (0,1),(0,2),(1,3),(2,3),(0,3) and #edges=5
print result

model = opengm.graphicalModel(states, operator="adder")
    
unary = np.array([vec, 1-vec]).T           # calculate values for unary terms
fid_u = model.addFunctions(unary)             # create a vector of functions for the value table
model .addFactors(fid_u,np.arange(0,vec.size)) # add those functions and the corresponding indices to the model

potts = opengm.pottsFunction(shape=[2,2],valueEqual=0.0,valueNotEqual=lam) # Define the potts func
fid_p = model.addFunction(potts)              # create a handle for the potts func
rows = img.shape[0]
cols = img.shape[1]

pairs, edge_num = gm.adjGraph(test1, False)
pairs.sort()

model.addFactors(fid_p,pairs)                # give the potts func and the pair list to the model

gc = opengm.inference.GraphCut(model)       
gc.infer()
result = gc.arg()