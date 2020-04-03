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
    for(size_t ie = 0; ie < num_faces; ie++ ) {
        for(size_t iv =0; iv < 3; iv++ ) {
            int nm1 = faces[ie*3+iv];
            vtoe[nm1*20 + nne[nm1]] = ie;
            nne[nm1] += 1;
            }
        }
    return vtoe;
}



// ----------------
// Python interface
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


PYBIND11_MODULE(cpputils, m) {
    m.def("vertex_to_elements", &vertex_to_elements);
}
