#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <compc/elias_gamma.hpp>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

py::array_t<uint8_t> compress(py::array_t<uint32_t> array){
  compc::EliasGamma<uint32_t> elias;
  const ssize_t* sizes = array.shape();
  std::size_t N = sizes[0];
  uint8_t* comp = elias.compress(array.data(), N);
  auto capsule = py::capsule(comp, [](void *v) {
    //std::cerr << "freeing memory @ " << v << std::endl;
    delete v;
  });
  return py::array(10, comp, capsule);
  
}


py::array_t<uint32_t> decompress(py::array_t<uint8_t> array){

  compc::EliasGamma<uint32_t> elias;
  const ssize_t* sizes = array.shape();
  std::size_t N = sizes[0];
  uint32_t* decomp = elias.decompress(array.data());
  auto capsule = py::capsule(decomp, [](void *v) {
    //std::cerr << "freeing memory @ " << v << std::endl;
    delete v;
  });
  return py::array(10, decomp, capsule);
}

PYBIND11_MODULE(comppy, m) {
    m.doc() = R"pbdoc(
        Fast Variable Length Encodings For Python
        -----------------------
        .. currentmodule:: fastvarints
        .. autosummary::
           :toctree: _generate
           compress
           decompress
    )pbdoc";


    m.def("compress", &compress, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m.def("decompress", &decompress, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
