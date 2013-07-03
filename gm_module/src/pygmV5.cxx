// boost python related
#include <boost/python/detail/wrap_python.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
#include <numpy/arrayobject.h>
#include <numpy/noprefix.h>

// vigra numpy array converters
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

// standart c++ headers (whatever you need (string and vector are just examples))
#include <string>
#include <vector>
#include <boost/unordered_set.hpp>
#include <boost/python/dict.hpp>


// this functions show how to loop over an
// entire image and do something with each pixel
// and store the result in a new image

struct edge
{
    uint32_t v1;
    uint32_t v2;
    edge(uint32_t sp1, uint32_t sp2) : v1(sp1), v2(sp2) {}
};


bool operator==(edge const& p1, edge const& p2)
{    return p1.v1 == p2.v1 && p1.v2 == p2.v2;}

std::size_t hash_value(edge const& p)
{
    std::size_t seed = 0;
    boost::hash_combine(seed, p.v1);
    boost::hash_combine(seed, p.v2);
    return seed;
}

boost::python::tuple adjGraph(vigra::NumpyArray<3,uint32_t> spixels, bool print)
{
    typedef vigra::NumpyArray<2,uint32_t> ArrayType; // what about bool and size_t arrays???
    typedef ArrayType::difference_type ArrayShape;

    size_t dimX=spixels.shape(0);
    size_t dimY=spixels.shape(1);
    size_t dimZ=spixels.shape(2);
    uint32_t sp1,sp2;
    size_t num_e = 0;
    boost::unordered_set<edge> edges; // hash table for storing edges: { key: sp 1 | value: sp 2 } and sp 1 > sp 2! Otherwise not unique!
    std::pair<boost::unordered::iterator_detail::c_iterator<const boost::unordered::detail::ptr_node<edge>*, boost::unordered::detail::ptr_node<edge>*, edge>, bool> insertion;
    // 3-fold scanning loop to detect neigbourhood-relations and mark them in the adjecancy-list/map
    for(size_t z=0; z<dimZ; ++z)
    {
        for(size_t y=0; y<dimY; ++y)
        {
            for(size_t x=0; x<dimX; ++x)
            {
                if(print) std::cout<<"x="<<int(x)<<" y="<<int(y)<<" z="<<int(z)<<": ";
                if((x<dimX-1) && (spixels(x,y,z) != spixels(x+1,y,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x+1,y,z));
                    sp2 = std::max(spixels(x,y,z),spixels(x+1,y,z));
                    if(print)std::cout<<"found ("<<sp1<<"|"<<sp2<<"), ";
                    insertion = edges.emplace(sp1, sp2);
                    if(insertion.second) num_e++;
                }
                if((y<dimX-1) && (spixels(x,y,z) != spixels(x,y+1,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x,y+1,z));
                    sp2 = std::max(spixels(x,y,z),spixels(x,y+1,z));
                    if(print)std::cout<<"found ("<<sp1<<"|"<<sp2<<"), ";
                    insertion = edges.emplace(sp1, sp2);
                    if(insertion.second) num_e++;
                }
                if((z<dimX-1) && (spixels(x,y,z) != spixels(x,y,z+1))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x,y,z+1));
                    sp2 = std::max(spixels(x,y,z),spixels(x,y,z+1));
                    if(print)std::cout<<"found ("<<sp1<<"|"<<sp2<<")";
                    insertion = edges.emplace(sp1, sp2);
                    if(insertion.second) num_e++;
                }
                if(print)std::cout<<std::endl;
            }
        }
    }

    if(print)std::cout<<std::endl;
    vigra::NumpyArray<2,uint32_t> edges_arr(ArrayShape(num_e,2));
    size_t i = 0;
    BOOST_FOREACH(edge e, edges)
    {
        if(print)std::cout<<"adding ("<<e.v1<<"|"<<e.v2<<")\n";
        edges_arr(i,0) = e.v1;
        edges_arr(i,1) = e.v2;
        i++;
    }
    return boost::python::make_tuple(edges_arr, num_e);
}

void export_adjGraph()
{
    // Do not change next 4 lines
    import_array();
    vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    // export the function to python
    boost::python::def("adjGraph", vigra::registerConverters(&adjGraph) ,
                       (
                           boost::python::arg("spixels, print")
                       ),
                       "Args:\n\n"
                       "   spixels : 3-d array of superpixels (SP), denoted by different uint32-values \n\n"
                       "   print   : turn on printing of processing information (for debugging)\n\n"
                       "returns: \n\n"
                       "   edges   : vector of adjacent SP (sp 1| sp 2) in a 6-neighbourhood. The number of the first SP is always > the number of the second SP\n\n"
                       "   #edges  : number of found edges\n\n"
                      );
}

boost::python::dict edgeFaces(vigra::NumpyArray<3,uint32_t> spixels, size_t num_e,  bool print)
{
    namespace bp = boost::python;
    //typedef vigra::NumpyArray<2,uint32_t> ArrayType; // what about bool and size_t arrays???
    //typedef ArrayType::difference_type ArrayShape;

    size_t dimX=spixels.shape(0);
    size_t dimY=spixels.shape(1);
    size_t dimZ=spixels.shape(2);

    uint32_t sp1,sp2,i1,i2;
    size_t key;
    bp::dict faces;

    // 3-fold scanning loop to detect neigbourhood-relations and mark them in the adjecancy-list/map
    for(size_t z=0; z<dimZ; ++z)
    {
        for(size_t y=0; y<dimY; ++y)
        {
            for(size_t x=0; x<dimX; ++x)
            {
                if(print) std::cout<<"x="<<int(x)<<" y="<<int(y)<<" z="<<int(z)<<": ";
                if((x<dimX-1) && (spixels(x,y,z) != spixels(x+1,y,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x+1,y,z));
                    sp2 = std::max(spixels(x,y,z),spixels(x+1,y,z));
                    i1  = x + dimX * x * y + dimX * x * dimY * y * z;
                    i2  = (x+1) + dimX * (x+1) * y + dimX * (x+1) * dimY * y * z;
                    if(print)std::cout<<"found ("<<sp1<<"|"<<sp2<<"), ";
                    key = sp1 * num_e + sp2;
                    if (!faces.has_key(key))
                    {
                        faces[key] = bp::list();
                        bp::extract<bp::list>(faces[key])().append(i1);
                    }
                    else bp::extract<bp::list>(faces[key])().append(i1);
                    bp::extract<bp::list>(faces[key])().append(i2);
                }
                if((y<dimX-1) && (spixels(x,y,z) != spixels(x,y+1,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x,y+1,z));
                    sp2 = std::max(spixels(x,y,z),spixels(x,y+1,z));
                    if(print)std::cout<<"found ("<<sp1<<"|"<<sp2<<"), ";
                    key = sp1 * num_e + sp2;
                    if (!faces.has_key(key))
                    {
                        faces[key] = bp::list();
                        bp::extract<bp::list>(faces[key])().append(i1);
                    }
                    else bp::extract<bp::list>(faces[key])().append(i1);
                    bp::extract<bp::list>(faces[key])().append(i2);
                }
                if((z<dimX-1) && (spixels(x,y,z) != spixels(x,y,z+1))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x,y,z+1));
                    sp2 = std::max(spixels(x,y,z),spixels(x,y,z+1));
                    if(print)std::cout<<"found ("<<sp1<<"|"<<sp2<<")";
                    key = sp1 * num_e + sp2;
                    if (!faces.has_key(key))
                    {
                        faces[key] = bp::list();
                        bp::extract<bp::list>(faces[key])().append(i1);
                    }
                    else bp::extract<bp::list>(faces[key])().append(i1);
                    bp::extract<bp::list>(faces[key])().append(i2);
                }
                if(print)std::cout<<std::endl;
            }
        }
    }
    return faces;
}

void export_edgeFaces()
{
    // Do not change next 4 lines
    import_array();
    vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    // export the function to python
    boost::python::def("edgeFaces", vigra::registerConverters(&edgeFaces) ,
                       (
                           boost::python::arg("spixels, print, #spixels")
                       ),
                       "Args:\n\n"
                       "   spixels : 3-d array of superpixels (SP), denoted by different uint32-values \n\n"
                       "   print   : turn on printing of processing information (for debugging)\n\n"
                       "   #spixels: Number of superpixels\n\n"
                       "returns: \n\n"
                       "   faces   : for each edge a collection (list) of facing voxels denoted by their absolute spacial index in the corresponding flattened array of the input\n\n"
                      );
}
