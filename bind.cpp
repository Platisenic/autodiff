#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <autodiff/autodiff.hpp>

namespace py = pybind11;
using namespace autodiff;


PYBIND11_MODULE(autodiff, m) {
    py::class_<Vector>(m, "vec")
        .def(py::init<size_t>())
        .def(py::init<std::vector<double> &>())
        .def("__len__", &Vector::size)
        .def("__getitem__", &Vector::getitem)
        .def("__setitem__", &Vector::setitem)
        .def("__repr__", &Vector::info)
        .def("grad", &Vector::grad)
        .def("values", &Vector::values)
        .def("backward", &Vector::backward)
        .def(py::self + py::self)
        .def(double() + py::self)
        .def(py::self + double())
        .def(py::self - py::self)
        .def(double() - py::self)
        .def(py::self - double())
        .def(py::self * py::self)
        .def(double() * py::self)
        .def(py::self * double())
        .def(py::self / py::self)
        .def(double() / py::self)
        .def(py::self / double())
        .def("sin", &Vector::sin)
        .def("cos", &Vector::cos)
        .def("tan", &Vector::tan)
        .def("exp", &Vector::exp)
        .def("log", &Vector::log)
        .def("sqrt", &Vector::sqrt)
        .def("abs", &Vector::abs);
}


