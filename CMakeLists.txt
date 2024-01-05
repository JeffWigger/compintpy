cmake_minimum_required(VERSION 3.4...3.18)
project(comppy)

add_subdirectory(include/compc compc)
target_compile_options(compc PRIVATE -fPIC)

add_subdirectory(pybind11)
pybind11_add_module(_comppy src/main.cpp)

target_link_libraries(_comppy PRIVATE compc)
# COMPPY_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_comppy
                           PRIVATE VERSION_INFO=${COMPPY_VERSION_INFO})
