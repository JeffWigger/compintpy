#include <pybind11/pybind11.h>

#include <compc/elias_gamma.hpp>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    std::size_t size = 10;
    long input[10] =  {1, 3, 2000, 2, 50, 1,25345, 11, 10000000, 1};
    compc::EliasGamma<long> elias;
    std::cout << "size: " << sizeof(input) << std::endl;
    uint8_t* comp = elias.compress(input, size);
    std::cout << "compressed" << std::endl;
    long* output = elias.decompress(comp);
    std::cout << "decompress" << std::endl;
    for(int i = 0; i < 10; i++){
        std::cout << output[i] << input[i] << std::endl; // comparing values
    }
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(comppy, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: comppy

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
