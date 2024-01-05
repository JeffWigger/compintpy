#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <compc/elias_gamma.hpp>
#include <compc/elias_delta.hpp>
#include <compc/elias_omega.hpp>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

template<typename T>
py::array_t<uint8_t> elias_gamma_compress(py::array_t<T, py::array::c_style> array, int offset, bool map_negative_numbers){
  compc::EliasGamma<T> elias(offset, map_negative_numbers);
  const ssize_t* sizes = array.shape();
  std::size_t N = sizes[0];
  uint8_t* comp = elias.compress(array.data(), N);
  auto capsule = py::capsule(comp, [](void *v) {
    // std::cerr << "freeing memory uint32 @ " << v << std::endl;
    delete [] static_cast<T*>(v);
  });
  return py::array(N, comp, capsule);
  
}

template<typename T>
py::array_t<T> elias_gamma_decompress(py::array_t<uint8_t> array, std::size_t binary_length, std::size_t array_length, int offset, bool map_negative_numbers){
  compc::EliasGamma<T> elias(offset, map_negative_numbers);
  T* decomp = elias.decompress(array.data(), binary_length, array_length);
  return py::array(array_length, decomp);
}


template<typename T>
py::array_t<uint8_t> elias_delta_compress(py::array_t<T, py::array::c_style> array, int offset, bool map_negative_numbers){
  compc::EliasDelta<T> elias(offset, map_negative_numbers);
  const ssize_t* sizes = array.shape();
  std::size_t N = sizes[0];
  uint8_t* comp = elias.compress(array.data(), N);
  auto capsule = py::capsule(comp, [](void *v) {
    // std::cerr << "freeing memory uint32 @ " << v << std::endl;
    delete [] static_cast<T*>(v);
  });
  return py::array(N, comp, capsule);
  
}

template<typename T>
py::array_t<T> elias_delta_decompress(py::array_t<uint8_t> array, std::size_t binary_length, std::size_t array_length, int offset, bool map_negative_numbers){
  compc::EliasDelta<T> elias(offset, map_negative_numbers);
  T* decomp = elias.decompress(array.data(), binary_length, array_length);
  return py::array(array_length, decomp);
}

template<typename T>
py::array_t<uint8_t> elias_omega_compress(py::array_t<T, py::array::c_style> array, int offset, bool map_negative_numbers){
  compc::EliasOmega<T> elias(offset, map_negative_numbers);
  const ssize_t* sizes = array.shape();
  std::size_t N = sizes[0];
  uint8_t* comp = elias.compress(array.data(), N);
  auto capsule = py::capsule(comp, [](void *v) {
    // std::cerr << "freeing memory uint32 @ " << v << std::endl;
    delete [] static_cast<T*>(v);
  });
  return py::array(N, comp, capsule);
  
}

template<typename T>
py::array_t<T> elias_omega_decompress(py::array_t<uint8_t> array, std::size_t binary_length, std::size_t array_length, int offset, bool map_negative_numbers){
  compc::EliasOmega<T> elias(offset, map_negative_numbers);
  T* decomp = elias.decompress(array.data(), binary_length, array_length);
  return py::array(array_length, decomp);
}


PYBIND11_MODULE(_comppy, m) {
    m.doc() = R"pbdoc(
        Fast Variable Length Intiger Encodings For Python
        -----------------------
        .. currentmodule:: fastvarints
        .. autosummary::
           :toctree: _generate
           compress
           decompress
    )pbdoc";


    // Gamma submodule
    auto m_g = m.def_submodule("gamma", "Elias Gamma functions");
    m_g.def("compress", &elias_gamma_compress<uint32_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_g.def("compress", &elias_gamma_compress<uint64_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_g.def("compress", &elias_gamma_compress<int32_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_g.def("compress", &elias_gamma_compress<int64_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_g.def("decompress_uint32", &elias_gamma_decompress<uint32_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_g.def("decompress_uint64", &elias_gamma_decompress<uint64_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_g.def("decompress_int32", &elias_gamma_decompress<int32_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_g.def("decompress_int64", &elias_gamma_decompress<int64_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");


    // Delta submodule
    auto m_d = m.def_submodule("delta", "Elias Delta functions");
    m_d.def("compress", &elias_delta_compress<uint32_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_d.def("compress", &elias_delta_compress<uint64_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_d.def("compress", &elias_delta_compress<int32_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_d.def("compress", &elias_delta_compress<int64_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_d.def("decompress_uint32", &elias_delta_decompress<uint32_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_d.def("decompress_uint64", &elias_delta_decompress<uint64_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_d.def("decompress_int32", &elias_delta_decompress<int32_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_d.def("decompress_int64", &elias_delta_decompress<int64_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");


    // Omega submodule
    auto m_o = m.def_submodule("omega", "Elias Delta functions");
    m_o.def("compress", &elias_omega_compress<uint32_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_o.def("compress", &elias_omega_compress<uint64_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_o.def("compress", &elias_omega_compress<int32_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_o.def("compress", &elias_omega_compress<int64_t>, py::return_value_policy::reference, R"pbdoc(
        compresses a numpy array using the elias gamma encoding
    )pbdoc");

    m_o.def("decompress_uint32", &elias_omega_decompress<uint32_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_o.def("decompress_uint64", &elias_omega_decompress<uint64_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_o.def("decompress_int32", &elias_omega_decompress<int32_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

    m_o.def("decompress_int64", &elias_omega_decompress<int64_t>, R"pbdoc(
        decompresses a numpy byte array with elias gamma encoded numbers
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
