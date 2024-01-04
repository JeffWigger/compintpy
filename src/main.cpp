#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <compc/elias_gamma.hpp>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

template<typename T>
py::array_t<uint8_t> compress(py::array_t<T, py::array::c_style> array){
  compc::EliasGamma<T> elias;
  const ssize_t* sizes = array.shape();
  std::size_t N = sizes[0];
  uint8_t* comp = elias.compress(array.data(), N);
  auto capsule = py::capsule(comp, [](void *v) {
    std::cerr << "freeing memory uint32 @ " << v << std::endl;
    delete [] static_cast<T*>(v);
  });
  return py::array(N, comp, capsule);
  
}

template<typename T>
py::array_t<T> decompress(py::array_t<uint8_t> array, std::size_t binary_length, std::size_t array_length){

  compc::EliasGamma<T> elias;
  T* decomp = elias.decompress(array.data(), binary_length, array_length);
  return py::array(array_length, decomp);
}

PYBIND11_MODULE(comppy, m) {
    m.doc() = R"pbdoc(
        Fast Variable Length Intiger Encodings For Python
        -----------------------
        .. currentmodule:: fastvarints
        .. autosummary::
           :toctree: _generate
           compress
           decompress
    )pbdoc";


    m.def("compress", &compress<uint32_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m.def("compress", &compress<uint64_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m.def("compress", &compress<int32_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m.def("compress", &compress<int64_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m.def("decompress_uint32", &decompress<uint32_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
