#ifndef LIBENTRY_H
#define LIBENTRY_H

#include <vector>
#include <any>
#include <unordered_set>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using KeyType = py::tuple;

namespace libentry {

class ArgProcessor {
public:
    ArgProcessor(int div) : divisibility_(div){};

    void classifyArguments(
        const py::list& args,
        const py::dict& kwargs,
        const py::list& jit_params,
        const std::unordered_set<int>& specialize_indices,
        const std::unordered_set<int>& do_not_specialize_indices);
    
    KeyType generateKey();

    py::list getKArgs();

private:
    py::list spec_args_; // specialize args
    py::list dns_args_; // do not specialize args
    py::list const_args_; // constexpr args
    py::list k_args_; // kernel args
    int divisibility_; // 对齐
};

} // namespace libentry

PYBIND11_MODULE(libentry_ascend, m) {
    py::class_<libentry::ArgProcessor>(m, "ArgProcessor")
        .def(py::init<int>())
        .def("classify_arguments", &libentry::ArgProcessor::classifyArguments,
            py::arg("args"),
            py::arg("kwargs"),
            py::arg("jit_params"),
            py::arg("specialize_indices"),
            py::arg("do_not_specialize_indices"),
            "classify arguments")
        .def("get_k_args", &libentry::ArgProcessor::getKArgs,
            "get kernel")
        .def("generate_key", &libentry::ArgProcessor::generateKey,
            "generate kernel cache key");
}

#endif