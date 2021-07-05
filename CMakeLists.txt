cmake_minimum_required (VERSION 3.3)
project (gibbs)

add_subdirectory(pybind11)
pybind11_add_module(gibbs src/gibbs.cpp)
