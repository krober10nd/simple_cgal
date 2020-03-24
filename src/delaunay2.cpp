#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <boost/lexical_cast.hpp>

#include <assert.h>    
#include <vector>


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel            Kernel;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, Kernel> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb>                       Tds;
typedef CGAL::Delaunay_triangulation_2<Kernel, Tds>                    Delaunay;
typedef Kernel::Point_2 Point;


std::vector<std::vector<int>> delaunay2(std::vector<double> &x, std::vector<double> &y) 
{
  int num_points = x.size();
  assert(y.size()!=num_points);
  std::vector< std::pair<Point,unsigned> > points;
  // add index information to form face table later
  for(std::size_t i = 0; i < num_points; ++i) 
  {
     points.push_back( std::make_pair( Point(x[i],y[i]), i ) );
  }

  Delaunay triangulation;
  triangulation.insert(points.begin(),points.end());

  // save the face table 
  int num_faces = triangulation.number_of_faces(); 
  std::vector<std::vector<int> > faces;
  faces.resize(3);
  for (int i = 0; i < 3; ++i)
    faces[i].resize(num_faces);

  int i=0;
  for(Delaunay::Finite_faces_iterator fit = triangulation.finite_faces_begin();
    fit != triangulation.finite_faces_end(); ++fit) {

    Delaunay::Face_handle face = fit;
    faces[0][i]=face->vertex(0)->info();
    faces[1][i]=face->vertex(1)->info();
    faces[2][i]=face->vertex(2)->info(); 
    //std::cout << face->vertex(0)->info() << " " << face->vertex(1)->info() << " " << face->vertex(2)->info() << std::endl;
    i+=1;
  }
  return faces;
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


