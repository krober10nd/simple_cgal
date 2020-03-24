#include <pybind11/pybind11.h>

#include <assert.h>    
#include <vector>


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_2<K>         Triangulation;
typedef Triangulation::Point             Point;

std::vector<int> delaunay2(std::vector<double> &x, std::vector<double> &y) {
  num_points = x.size();
  assert(y.size()!=num_points);
  Triangulation t;
  for(std::size_t i = 0; i < num_points; i++) {
     t.insert(Point(x[i],y[i]);
    }
  return 0;
}

namespace py = pybind11;

PYBIND11_MODULE(simple_cgal, m) {
    m.def("delaunay2", &delaunay2);
}
