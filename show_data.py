import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plot
import numpy
import h5py
import sys
import vigra

if __name__ == "__main__":
    f = h5py.File("data/block01.h5", 'r')
    d = f["volume/data"].value
    print d.shape

    d = d[0:60,0:60,0:60]

    ws, maxRegionLabel = vigra.analysis.watersheds(d.astype(numpy.uint8))

    g = h5py.File("ws.h5", 'w')
    g.create_dataset("ws", data=ws)
    g.close()

    plot.figure()
    plot.gray()
    plot.imshow(d[:,:,30])

    plot.figure()

    ctable = numpy.random.random((ws.max()+1, 3))
    print ws.shape
    print ctable.shape
    c = ctable[ws]
    print c.shape

    plot.imshow(c[:,:,30])
    plot.show()

    print ws.shape
