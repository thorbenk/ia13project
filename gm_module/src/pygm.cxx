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
#include <boost/unordered_map.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>


// this functions show how to loop over an
// entire image and do something with each pixel
// and store the result in a new image


//typedef vigra::NumpyArray<2,uint32_t> ArrayType;
//typedef ArrayType::difference_type ArrayShape;

// vigra::NumpyArray<2,uint32_t>::difference_type

struct edge // key type for map
{
    uint32_t v1;
    uint32_t v2;
    edge(uint32_t sp1, uint32_t sp2) : v1(sp1), v2(sp2) {}
};

struct edgeData // for each edge store certain features/data
{
    size_t volume;
    //size_t circumference; not implemented
    float  avg_value;
    float  avg_gradient;
    float  std_value;
    float  std_gradient;
    vigra::NumpyArray<2,uint32_t> face;
    //size_t size1; not implemented
    //size_t size2; not implemented

    size_t counter;
    edgeData() : volume(0), avg_gradient(0), avg_value(0), std_gradient(0), std_value(0), counter(0) {} // initialise all to zero
};
bool operator==(edge const& p1, edge const& p2) // speicify equality-relation for keys
{
    return p1.v1 == p2.v1 && p1.v2 == p2.v2;
}

std::size_t hash_value(edge const& p) // overload hash-fuction for custom key-type
{
    std::size_t seed = 0;
    boost::hash_combine(seed, p.v1);
    boost::hash_combine(seed, p.v2);
    return seed;
}

typedef boost::unordered_map<edge,edgeData> EdgeMapType;

boost::python::tuple adjGraph(vigra::NumpyArray<3,uint32_t> spixels,
                              vigra::NumpyArray<3, float>   data,
                              bool facelist)
{
    size_t dimX=spixels.shape(0);
    printf("dimX=%d\t", int(dimX));
    size_t dimY=spixels.shape(1);
    printf("dimY=%d\t", int(dimY));
    size_t dimZ=spixels.shape(2);
    printf("dimZ=%d\n", int(dimZ));
    uint32_t sp1,sp2;
    edge key(0,0);
    size_t num_e = 0;
    EdgeMapType edges; // hash table for storing edges: { key: sp 1 | value: sp 2 } and sp 1 > sp 2! Otherwise not unique!
    // 3-fold scanning loop to detect neigbourhood-relations and mark them in the adjecancy-list/map
    for(size_t z=0; z<dimZ; ++z)
    {
        for(size_t y=0; y<dimY; ++y)
        {
            for(size_t x=0; x<dimX; ++x)
            {
                if((x<dimX-1) && (spixels(x,y,z) != spixels(x+1,y,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x+1,y,z)); // use the smaller sp-id as the first key
                    sp2 = std::max(spixels(x,y,z),spixels(x+1,y,z)); // the greater as the second key

                    key = edge(sp1,sp2);                // create an key object
                    if (edges.find(key) == edges.end()) // if there is no edge for this key in the dictionary
                    {
                        edges[key] = edgeData();        // create a new edge in dictionary
                        num_e++;                        // increment number of edges
                    }
                    edges[key].volume += 2;             // in any case increment the number of pixels in the edge by two (we regard the whole pair of pixels)
                }
                if((y<dimY-1) && (spixels(x,y,z) != spixels(x,y+1,z)))
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x,y+1,z));
                    sp2 = std::max(spixels(x,y,z),spixels(x,y+1,z));
                    key = edge(sp1,sp2);
                    if (edges.find(key) == edges.end())
                    {
                        edges[key] = edgeData();
                        num_e++;
                    }
                    edges[key].volume += 2;
                }
                if((z<dimZ-1) && (spixels(x,y,z) != spixels(x,y,z+1)))
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x,y,z+1));
                    sp2 = std::max(spixels(x,y,z),spixels(x,y,z+1));
                    key = edge(sp1,sp2);
                    if (edges.find(key) == edges.end())
                    {
                        edges[key] = edgeData();
                        num_e++;
                    }
                    edges[key].volume += 2;
                }
            }
        }
    }

    if(facelist) // if the caller requires the return of the face list
    {
        BOOST_FOREACH(EdgeMapType::value_type &p, edges)
        {
            p.second.face = vigra::NumpyArray<2,uint32_t>(vigra::NumpyArray<2,uint32_t>::difference_type(p.second.volume,3));  // Initialise coordinate list of pixels to correct lenght and for 3 coordinates
        }
    }

    edgeData * value; // this points to the value found in the dictionary
    for(size_t z=0; z<dimZ; ++z)
    {
        for(size_t y=0; y<dimY; ++y)
        {
            for(size_t x=0; x<dimX; ++x)
            {
                if((x<dimX-1) && (spixels(x,y,z) != spixels(x+1,y,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x+1,y,z));  // use the smaller sp-id as the first key
                    sp2 = std::max(spixels(x,y,z),spixels(x+1,y,z));  // the greater as the second key
                    key = edge(sp1,sp2);                              // create key object
                    value = &(edges.find(key)->second);               // lookup value in dictionary. (return of find is a tuple, therfore ->second. Because the scanning is the same as in the previous loop, it is not possible that there is no lookup-value for a key, otherwise boost fails or sth...)

                    if(facelist)                                // if the caller requires the return of the face list
                    {
                        value->face(value->counter, 0) = x;     // insert the x-coord of the first pixel in the array. The counter is used to find the next free line.
                        value->face(value->counter, 1) = y;     // insert the y-coord...
                        value->face(value->counter, 2) = z;
                        value->counter++;                       // now a complete point was added -> increment the counter
                        value->face(value->counter, 0) = x+1;;  // add the x-coordinate of the neighbouring point in the next line in the array
                        value->face(value->counter, 1) = y;     // ...
                        value->face(value->counter, 2) = z;
                        value->counter++;
                    }

                    value->avg_value    += (data(x,y,z) + data(x+1,y,z)) / float(value->volume);        // add the contribution of this pixel pair to the average value on the edge
                    value->avg_gradient += 2*abs(data(x,y,z) - data(x+1,y,z)) / float(value->volume);   // also add the gradient contribution, factor two is needed, because for 2N pixels we can only caluculate N gradients
                }
                if((y<dimY-1) && (spixels(x,y,z) != spixels(x,y+1,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x,y+1,z));
                    sp2 = std::max(spixels(x,y,z),spixels(x,y+1,z));
                    key = edge(sp1,sp2);
                    value = &(edges.find(key)->second);

                    if(facelist)
                    {
                        value->face(value->counter, 0) = x;
                        value->face(value->counter, 1) = y;
                        value->face(value->counter, 2) = z;
                        value->counter++;
                        value->face(value->counter, 0) = x;
                        value->face(value->counter, 1) = y+1;
                        value->face(value->counter, 2) = z;
                        value->counter++;
                    }

                    value->avg_value    += (data(x,y,z) + data(x,y+1,z)) / float(value->volume);
                    value->avg_gradient += 2*abs(data(x,y,z) - data(x,y+1,z)) / float(value->volume);
                }
                if((z<dimZ-1) && (spixels(x,y,z) != spixels(x,y,z+1))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp1 = std::min(spixels(x,y,z),spixels(x,y,z+1));
                    sp2 = std::max(spixels(x,y,z),spixels(x,y,z+1));
                    key = edge(sp1,sp2);
                    value = &(edges.find(key)->second);
                    if(facelist)
                    {
                        value->face(value->counter, 0) = x;
                        value->face(value->counter, 1) = y;
                        value->face(value->counter, 2) = z;
                        value->counter++;
                        value->face(value->counter, 0) = x;
                        value->face(value->counter, 1) = y;
                        value->face(value->counter, 2) = z+1;
                        value->counter++;
                    }

                    value->avg_value    += (data(x,y,z) + data(x,y,z+1)) / float(value->volume);
                    value->avg_gradient += 2*abs(data(x,y,z) - data(x,y,z+1)) / float(value->volume);
                }
            }
        }
    }

    vigra::NumpyArray<2,float> edges_arr;       // Create new array to return the edges' vertices and their features compactly
    edges_arr.reshape(vigra::Shape2(num_e,5));
    boost::python::list face_list;              // The arrays with the coordinates of the surface pixels are appended to a list
    size_t i = 0;
    BOOST_FOREACH(EdgeMapType::value_type p, edges)
    {
        edges_arr(i,0) = p.first.v1;           // first vertice of edge
        edges_arr(i,1) = p.first.v2;           // second vertice of edge
        edges_arr(i,2) = p.second.volume;      // and the features...
        edges_arr(i,3) = p.second.avg_value;
        edges_arr(i,4) = p.second.avg_gradient;
        if(facelist) face_list.append(p.second.face);
        i++;
    }
    if(facelist) return boost::python::make_tuple(edges_arr,face_list);
    else return boost::python::make_tuple(edges_arr);
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
                           boost::python::arg("spixels, data, getsurfaces")
                       ),
                       "Args:\n\n"
                       "   spixels    : 3-d array of superpixels (SP), denoted by different uint32-values, NumpyArray \n\n"
                       "   data       : raw image data to compute edgewise features float32-NumpyArray\n\n"
                       "   getsurfaces: BOOL, turn on to get the second return argument, if turned off, the return value is still a tuple containing only on element\n\n"
                       "returns: \n\n"
                       "   edges     : array of adjacent SP in a 6-neighbourhood and their features in the form: (sp1| sp2 | volume | average | gradient ) . The number of the first SP is always > the number of the second SP.\n\n"
                       "               * volume   : number of pixels in the facing surface\n"
                       "               * value    : average data value on the surface\n"
                       "               * gradient : average (absolute) gradient across face\n\n"
                       "   surfaces : list with arrays of coordinates of the points which lie on a facing surface. The list contains an array per edge and the order in the list is the same as in the edges-output. Note: this is the only way to determine where the points of each list entry belong to!\n\n"
                      );
}
