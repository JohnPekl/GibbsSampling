#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gibbs.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gibbs, m) {
    m.def("gibbs_jointpredupdt", &gibbs_jointpredupdt);
}
