#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <assert.h>
#include <vector>




std::vector<int> c_vertex_to_elements(std::vector<int> &faces, int &num_points, int &num_faces)
{
    std::vector<int> vtoe;
    std::vector<int> nne;
    // assume each vertex has a max. of 20 elements neigh.
    vtoe.resize(num_points*20);
    nne.resize(num_points);
    for(std::size_t ie = 0; ie < num_faces; ie++ ) {
        for(std::size_t iv =0; iv < 3; iv++ ) {
            int nm1 = faces[ie*3+iv];
            vtoe[nm1*20 + nne[nm1]] = ie;
            nne[nm1] += 1;
            }
        }
    return vtoe;
}



// ----------------
// Python interface for vertex_to_elements
// ----------------
// (from https://github.com/tdegeus/pybind11_examples/blob/master/04_numpy-2D_cpp-vector/example.cpp)

namespace py = pybind11;
py::array vertex_to_elements(py::array_t<int, py::array::c_style | py::array::forcecast> faces,
        int num_points, int num_faces)
{

  // check input dimensions
  if ( faces.ndim() != 2 )
    throw std::runtime_error("Input should be a 2D NumPy array");

  if ( num_points < 3 )
      throw std::runtime_error("Too few points!");

  if (num_faces != faces.size()/3)
      throw std::runtime_error("Number of faces doesn't match!");

  // allocate std::vector (to pass to the C++ function)
  std::vector<int> cppfaces(num_faces*3);

  // copy py::array -> std::vector
  std::memcpy(cppfaces.data(),faces.data(),num_faces*3*sizeof(int));
  std::vector<int> vtoe = c_vertex_to_elements(cppfaces, num_points, num_faces);

  ssize_t              soint      = sizeof(int);
  ssize_t              ndim      = 2;
  std::vector<ssize_t> shape     = {num_points, 20};
  std::vector<ssize_t> strides   = {soint*20, soint};

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    vtoe.data(),                           /* data as contiguous array  */
    sizeof(int),                          /* size of one scalar        */
    py::format_descriptor<int>::format(), /* data type                 */
    2,                                    /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));
}





bool c_do_intersect2(std::vector<double> &c, double &r,
        std::vector<double> &le, std::vector<double> &re) {

    for(std::size_t i =0; i < 2; i++) {
        if(c[i] < le[i]){
            if(c[i] + r < le[i]){
                return false;
            }
        }
        else if(c[i] > re[i]) {
            if(c[i] - r > re[i]){
                return false;
            }
        }
    }
    return true;
}




// ----------------
// Python interface for do_intersect2
// ----------------

bool do_intersect2(
        py::array_t<double, py::array::c_style | py::array::forcecast> c,
        double r,
        py::array_t<double, py::array::c_style | py::array::forcecast> le,
        py::array_t<double, py::array::c_style | py::array::forcecast> re
        )
{
  // check input dimensions
  if ( c.ndim() != 1 )
    throw std::runtime_error("Input should be a 1D NumPy array");
  if ( le.ndim() != 1 )
    throw std::runtime_error("Input should be a 1D NumPy array");
  if ( re.ndim() != 1 )
    throw std::runtime_error("Input should be 1D NumPy array");

  // allocate std::vector (to pass to the C++ function)
  std::vector<double> cppC(2);
  std::vector<double> cppLE(2);
  std::vector<double> cppRE(2);
  double cppR;

  // copy py::array -> std::vector
  std::memcpy(cppLE.data(),re.data(),2*sizeof(double));
  std::memcpy(cppRE.data(),le.data(),2*sizeof(double));
  std::memcpy(cppC.data(),c.data(),2*sizeof(double));
  cppR = r;

  bool do_intersect = c_do_intersect2(cppC, cppR, cppLE, cppRE);

  // return result
  return do_intersect;
}




std::vector<int> c_sph_bx_intersect2(std::vector<double> &circumcenters,
                                    std::vector<double> &radii,
                                    std::vector<double> &le,
                                    std::vector<double> &re) {

    int num_circumcenters = radii.size();
    std::vector<int> intersects;
    intersects.resize(num_circumcenters*2);

    for(std::size_t i=0; i < num_circumcenters; ++i)
    {
        // two neighboring blocks: below and above (in this order)
        for(std::size_t j=0; j < 2; ++j)
        {
            bool do_intersect = true;
            // loop over dimensions
            for(std::size_t n =0; n < 2; n++) {
                if(circumcenters[2*i + n] < le[j*2 + n]){
                    if((circumcenters[2*i + n] + radii[i]) < le[j*2 + n]){
                        do_intersect=false;
                    }
                }
                else if(circumcenters[2*i + n] > re[j*2 + n]) {
                    if((circumcenters[2*i + n] - radii[i]) > re[j*2 + n]){
                        do_intersect=false;
                    }
                }
            }
            if(do_intersect) {
                // 0 for block below
                // and 1 for block above
                intersects[i*2 + j] = j;
                }
            else {
                intersects[i*2 + j] = -1;
            }
        }
    }
    return intersects;
}


// ----------------
// Python interface for c_sph_bx_intersect2
// ----------------

 py::array sph_bx_intersect2(
            py::array_t<double, py::array::c_style | py::array::forcecast> circumcenters,
            py::array_t<double, py::array::c_style | py::array::forcecast> radii,
            py::array_t<double, py::array::c_style | py::array::forcecast> le,
            py::array_t<double, py::array::c_style | py::array::forcecast> re)
{
  int num_circumcenters = radii.size();

  // allocate std::vector (to pass to the C++ function)
  std::vector<double> cppCC(2*num_circumcenters);
  std::vector<double> cppRR(num_circumcenters);
  std::vector<double> cppLE(4);
  std::vector<double> cppRE(4);

  // copy py::array -> std::vector
  std::memcpy(cppRE.data(),re.data(),4*sizeof(double));
  std::memcpy(cppLE.data(),le.data(),4*sizeof(double));
  std::memcpy(cppCC.data(),circumcenters.data(),num_circumcenters*2*sizeof(double));
  std::memcpy(cppRR.data(),radii.data(),num_circumcenters*sizeof(double));

  std::vector<int> intersect = c_sph_bx_intersect2(cppCC, cppRR, cppLE, cppRE);

  ssize_t              soint      = sizeof(int);
  ssize_t              ndim      = 2;
  std::vector<ssize_t> shape     = {num_circumcenters, 2};
  std::vector<ssize_t> strides   = {soint*2, soint};

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    intersect.data(),                           /* data as contiguous array  */
    sizeof(int),                          /* size of one scalar        */
    py::format_descriptor<int>::format(), /* data type                 */
    2,                                    /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));
}


PYBIND11_MODULE(cpputils, m) {
    m.def("vertex_to_elements", &vertex_to_elements);
    m.def("do_intersect", &do_intersect2);
    m.def("sph_bx_intersect2", &sph_bx_intersect2);
}
