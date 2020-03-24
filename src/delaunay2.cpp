#include <pybind11/pybind11.h>

#include <assert.h>    
#include <vector>


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> DT;
typedef DT::Point Point;

std::vector<double> delaunay2(std::vector<double> &x, std::vector<double> &y) 
{
  int num_points = x.size();
  assert(y.size()!=num_points);
  DT t;
  for(std::size_t i = 0; i < num_points; i++) {
     t.insert(Point(x[i],y[i]));
    }
  return x;
}

namespace py = pybind11;

PYBIND11_MODULE(simple_cgal, m) {
    m.def("delaunay2", &delaunay2);
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}


