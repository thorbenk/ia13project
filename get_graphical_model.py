import opengm
import numpy



def getGraphicalModel(
    nLabels,
    nVariables,
    nEdges,
    edgeVis,
    unaryValues,
    edgeValues,
    gmOperator='adder'
):
    """ get opengm graphical model
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!! USE OPENGM FORK https://github.com/DerThorsten/opengm  !!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    kwargs :

        nLabels         :   number of labels / classes of the gm

        nVariables      :   number of variables / superpixels of the gm

        nEdges          :   number of edges / 2.-order factors

        edgeVis         :   a 2d numpy array of the variable indices of 
                            an edge / 2.-order factor with shape=[nEdges,2].
                            The variable indices must be sorted:
                            Therefore edgeVis[:,0]<edgeVis[:,1] must be true

        unaryValues     :   unaries in a 2d numpy array with shape=[nVariables,nLabels]

        edgeValues      :   2d numpy array for values edges / 2.-order factors.
                            For each 2.-order factor one needs only 2 values.

        gmOperator      :   operator of the graphical model 'adder' or 'multiplier' 
                            (default: 'adder')

    """


    #####################################################################################
    # assertions to check that the input is valid
    #####################################################################################

    def raiseIfNot(cond,msg):
        if(cond==False):
            raise RuntimeError(msg+" is NOT true")

    raiseIfNot(unaryValues.ndim==2,"unaryValues.ndim== 2")
    raiseIfNot(unaryValues.shape[0]==nVariables,"unaryValues.shape[0]== nVariables")
    raiseIfNot(unaryValues.shape[1]==nLabels,"unaryValues.shape[1]==nLabels")
    raiseIfNot(edgeVis.ndim==2,"edgeVis.ndim== 2")
    raiseIfNot(edgeVis.shape[0]==nEdges,"edgeVis.shape[0]==nEdges")
    raiseIfNot(edgeVis.shape[1]==2,"edgeVis.shape[1]==2")
    raiseIfNot ( len(numpy.where(edgeVis[:,0]>=edgeVis[:,1])[0] )==0 ,
            "edgeVis[:,0]<edgeVis[:,1]" )
    raiseIfNot(edgeValues.ndim==2,"edgeValues.ndim== 2")
    raiseIfNot(edgeValues.shape[0]==nEdges,"edgeValues.shape[1]==nEdges")
    raiseIfNot(edgeValues.shape[1]==2,"edgeValues.shape[1]==2")
    

    #####################################################################################
    # set up space of graphical model and construct a opengm.gm
    #####################################################################################

    numberOfLabels = numpy.ones(nVariables,dtype=opengm.label_type)*nLabels
    gm             = opengm.gm(numberOfLabels,operator=gmOperator)


    #####################################################################################
    # reserve space for factors and functions
    #####################################################################################

    gm.reserveFactors(nVariables + nEdges)
    # reserve explicit functions for unaries
    gm.reserveFunctions(nVariables,'explicit')
    # reserve potts functions for 2-order factors
    gm.reserveFunctions(nEdges,'potts')


    #####################################################################################
    # add unary functions and factors to graphical model
    #####################################################################################
    
    # add unary functions (and check for consistency)
    fidUnaries = gm.addFunctions(unaryValues)
    raiseIfNot(len(fidUnaries)==nVariables,"internal error, blame thorsten")

    # add unary factors (and check for consistency)
    unaryVis = numpy.arange(nVariables,dtype=opengm.index_type)
    gm.addFactors(fidUnaries,unaryVis)
    raiseIfNot(gm.numberOfFactors==nVariables,"internal error, blame thorsten")

    #####################################################################################
    # add 2-order functions and factors
    #####################################################################################

    # create a vector of potts functions (and check for consistency)
    valueAA = edgeValues[:,0]   #  F(L_1 == L_2)  
    valueAB = edgeValues[:,1]   #  F(L_1 != L_2)
    pottsFunctions = opengm.pottsFunctions( shape=[nLabels,nLabels],
                                            valueEqual=valueAA,valueNotEqual=valueAB)
    raiseIfNot(len(pottsFunctions)==nEdges,"internal error, blame thorsten")

    # add second order potts functions (and check for consistency)
    fid2Order = gm.addFunctions(pottsFunctions)
    raiseIfNot(len(fid2Order)==nEdges,"internal error, blame thorsten")

    # add second order factors (and check for consistency)
    gm.addFactors(fid2Order,edgeVis)
    raiseIfNot(gm.numberOfFactors==nVariables+nEdges,"internal error, blame thorsten")


    return gm



    



if __name__ == "__main__":

    """
    simple 2x2 grid
    _______ 
    |0 | 1|
    |__.__|
    |2 | 3|
    |__|__|
    """

    nLabels     = 3
    nVariables  = 4
    nEdges      = 4
    edgeVis     = numpy.array([[0,1],[2,3],[0,2],[1,3]])
    unaryValues = numpy.random.rand(nVariables,nLabels)
    edgeValues  = numpy.random.rand(nEdges,2)
    gmOperator  = 'adder'


    gm = getGraphicalModel(
        nLabels     = nLabels,
        nVariables  = nVariables,
        nEdges      = nEdges,
        edgeVis     = edgeVis,
        unaryValues = unaryValues,
        edgeValues  = edgeValues,
        gmOperator  = 'adder'
    )

    print gm