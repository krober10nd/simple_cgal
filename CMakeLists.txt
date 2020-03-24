cmake_minimum_required(VERSION 2.8.12)
project(simple_cgal)

add_subdirectory(pybind11)
pybind11_add_module(simple_cgal src/delaunay2.cpp)
