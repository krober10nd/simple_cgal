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

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned int, Kernel> Vb3;
typedef CGAL::Triangulation_data_structure_3<Vb3>                       Tds3;
typedef CGAL::Delaunay_triangulation_3<Kernel, Tds3>                    Delaunay3;
typedef Kernel::Point_3 Point3; 

std::vector<std::vector<int>> delaunay3(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z) 
{
  int num_points = x.size();
  assert(y.size()!=num_points);
  assert(z.size()!=num_points);
  std::vector< std::pair<Point3,unsigned> > points;
  // add index information to form face table later
  for(std::size_t i = 0; i < num_points; ++i) 
  {
     points.push_back( std::make_pair( Point3(x[i],y[i],z[i]), i ) );
  }
  Delaunay3 triangulation;
  triangulation.insert(points.begin(),points.end());
  // save the indices of all cells
  int num_cells = triangulation.number_of_finite_cells(); 
  std::vector<std::vector<int> > cells;
  cells.resize(4);
  for (int i = 0; i < 4; ++i)
    cells[i].resize(num_cells);
  std::cout << num_cells << std::endl;

  int i=0;
  for(Delaunay3::Finite_cells_iterator cit = triangulation.finite_cells_begin();
    cit != triangulation.finite_cells_end(); ++cit) {

    Delaunay3::Cell_handle cell = cit;
    cells[0][i]=cell->vertex(0)->info();
    cells[1][i]=cell->vertex(1)->info();
    cells[2][i]=cell->vertex(2)->info(); 
    cells[3][i]=cell->vertex(3)->info(); 
    i+=1;
  }
  return cells;
}


PYBIND11_MODULE(simple_cgal, m) {
    m.def("delaunay2", &delaunay2);
    m.def("delaunay3", &delaunay3);
}

