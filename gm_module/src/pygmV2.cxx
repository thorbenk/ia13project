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

// my headers  ( in my_module/include )
#include <my_include.hxx>


// this functions show how to loop over an
// entire image and do something with each pixel
// and store the result in a new image

vigra::NumpyAnyArray  imageLoop(

    vigra::NumpyArray<2,float> inputImage
){

    // allocate output image (also a float image in this case)
    typedef vigra::NumpyArray<2,float> ArrayType;
    typedef ArrayType::difference_type ArrayShape;
    size_t dimX=inputImage.shape(0);
    size_t dimY=inputImage.shape(1);

    ArrayType resultArray(ArrayShape(dimX,dimY));

    for(size_t y=0;y<dimY;++y){
        for(size_t x=0;x<dimX;++x){

            // do *something* a pixel  and store in in the result array
            resultArray(x,y)=2*inputImage(x,y)+1;
        }
    }

    return resultArray;
}



boost::python::tuple adjGraph(vigra::NumpyArray<3,float> classes)
//vigra::NumpyArray adjGraph(vigra::NumpyArray<3,float> classes)
{
    typedef vigra::NumpyArray<2,float> ArrayType; // what about bool and size_t arrays???
    typedef ArrayType::difference_type ArrayShape;

    size_t dimX=classes.shape(0);
    size_t dimY=classes.shape(1);
    size_t dimZ=classes.shape(2);

    size_t pix_id; // unique number to label pixel as set member
    size_t nei_id; // and for neigbour

    vigra::NumpyArray<3,size_t> sp_map(ArrayShape(dimX,dimY,dimZ));   // a mapping array that associsates a pixels ide (col 1) to a representative id (col 2)
    size_t sp_num = 0;
    // 3-fold scanning loop to map superpixels to representative <=> number all contiguous super pixels (in 6 neigbourhood) ascendingly while traversing the cube
    for(size_t z=0;z<dimZ;++z)
    {
        for(size_t y=0;y<dimY;++y)
        {
            for(size_t x=0;x<dimX;++x)
            {
                if((x>=0) && (classes(x,y,z) == classes(x-1,y,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    sp_map(x,y,z) = sp_map(x-1,y,z);              // if the pixelx have identical classifications, set rep.element of current pixel to rep of pixel of same class
                }
                else if((y>=0) && (classes(x,y,z) == classes(x,y-1,z)))
                {
                    sp_map(x,y,z) = sp_map(x,y-1,z);
                }
                else if((z>=0) && (classes(x,y,z) == classes(x,y,z-1)))
                {
                    sp_map(x,y,z) = sp_map(x,y,z-1);
                }
                else                                              // is either the first occurance of super pixel or there no pixel of same class around (yet)
                {
                    sp_map(x,y,z) = sp_num;                       // assign a new supepixel number
                    sp_num++;                                     // increment the sp-num, ate this also gives the total number of superpixels
                }
            }
        }
    }


    vigra::NumpyArray<2,bool> adj_mat(ArrayShape(sp_num, sp_num));
    size_t e_num = 0;
    // 3-fold scanning loop to detect neigbourhood-relations and mark them in the adjecancy-matrix
    for(size_t z=0;z<dimZ;++z)
    {
        for(size_t y=0;y<dimY;++y)
        {
            for(size_t x=0;x<dimX;++x)
            {
                if((x<dimX-1) && (classes(x,y,z) != classes(x+1,y,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    pix_id = sp_map(x,y,z);          // find sp_id for current pixel
                    nei_id = sp_map(x+1,y,z);        // and for neigbour
                    if(pix_id <= nei_id) adj_mat(pix_id, nei_id) = true;  // because those two different superpixels are neigbours, mark them in adj_matrix
                    else adj_mat(nei_id, pix_id) = true;                  // only fill upper triangle of adj-mat
                }
                if((y<dimX-1) && (classes(x,y,z) != classes(x,y+1,z))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    pix_id = sp_map(x,y,z);          // find sp_id for current pixel
                    nei_id = sp_map(x,y+1,z);        // and for neigbour
                    if(pix_id <= nei_id) adj_mat(pix_id, nei_id) = true;  // because those two different superpixels are neigbours, mark them in adj_matrix
                    else adj_mat(nei_id, pix_id) = true;                  // only fill upper triangle of adj-mat
                }
                if((z<dimX-1) && (classes(x,y,z) != classes(x,y,z+1))) // 1. check bounds 2. if two adjecant voxel have different classification
                {
                    pix_id = sp_map(x,y,z);          // find sp_id for current pixel
                    if(pix_id <= nei_id) adj_mat(pix_id, nei_id) = true;  // because those two different superpixels are neigbours, mark them in adj_matrix
                    else adj_mat(nei_id, pix_id) = true;                  // only fill upper triangle of adj-mat
                }
            }
        }
    }

    vigra::NumpyArray<2,size_t> edges(ArrayShape(e_num,2));
    for(size_t i=0;i<sp_num;++i) /// y
    {
        for(size_t j=0;j<=i;++j) /// x only traverse through one triangle of adj-matrix because it is symmetric
        {
            if adj_mat(j,i)
            {
                edges() = j;
                edges() = i;
            }
        }
    }
    return boost::python::make_tuple(edges, sp_map, sp_num)
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
            boost::python::arg("classes")
        ),
        "loop over an image and do something with each pixels\n\n"
        "Args:\n\n"
        "   classes : 3-d array of classes, denoted by different ints \n\n"
        "returns: \n\n"
        "   superpixels: list of tupels (superpixel number | representative pixel). Representant is the first pixel if the cube is traversed in <> order\n"
        "   neighbours : list of tupels (superpixel 1 | superpixel 2) if 1,2 are neighbours in a 6-neighbourhood\n\n"
    );
}


void export_HelloWorld(){
    // Do not change next 4 lines
    import_array();
    vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    // export the function to python
    boost::python::def("imageLoop", vigra::registerConverters(&imageLoop) ,
        (
            boost::python::arg("image")
        ),
        "loop over an image and do something with each pixels\n\n"
        "Args:\n\n"
        "   image : input image\n\n"
        "returns an an image with the same shape as the input image"
    );
}
