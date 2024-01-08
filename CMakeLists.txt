cmake_minimum_required(VERSION 3.4...3.18)
project(compintpy)

add_subdirectory(include/compc compintc)
target_compile_options(compintc PRIVATE -fPIC)

add_subdirectory(pybind11)
pybind11_add_module(_compintpy src/main.cpp)

target_link_libraries(_compintpy PRIVATE compintc)

target_compile_definitions(_compintpy
                           PRIVATE VERSION_INFO=${compintpy_VERSION_INFO})
