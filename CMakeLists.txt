cmake_minimum_required(VERSION 3.4...3.18)
project(comppy)

add_subdirectory(include/compc compc)


add_subdirectory(pybind11)
pybind11_add_module(comppy src/main.cpp)

target_link_libraries(comppy PRIVATE compc)
# COMPPY_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(comppy
                           PRIVATE VERSION_INFO=${COMPPY_VERSION_INFO})
