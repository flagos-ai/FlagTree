#include "runtime/libentry/libentry.h"

using namespace libentry;

void libentry::ArgProcessor::classifyArguments(
    const py::list& args,
    const py::dict& kwargs,
    const py::list& jit_params,
    const std::unordered_set<int>& specialize_indices,
    const std::unordered_set<int>& do_not_specialize_indices)
{
    for (size_t i = 0; i < args.size(); ++i) {
        if (specialize_indices.count(i)) {
            k_args_.append(args[i]);
            spec_args_.append(args[i]);
        } else if (do_not_specialize_indices.count(i)) {
            k_args_.append(args[i]);
            dns_args_.append(args[i]);
        } else {
            const_args_.append(args[i]);
        }
    }

    for (size_t i = args.size(); i < jit_params.size(); ++i) {
        const py::object& param = jit_params[i];
        py::object val;

        if (kwargs.contains(param.attr("name"))) {
            val = kwargs[param.attr("name")];
        } else if (py::hasattr(param, "default") && !param.attr("default").is_none()) {
            val = param.attr("default");
        } else {
            continue;
        }

        if (param.attr("is_constexpr").cast<py::bool_>()) {
            const_args_.append(val);
        } else if (param.attr("do_not_specialize").cast<py::bool_>()) {
            dns_args_.append(val);
            k_args_.append(val);
        } else {
            spec_args_.append(val);
            k_args_.append(val);
        }
    }
}

KeyType libentry::ArgProcessor::generateKey()
{
    auto is_tensor = [](py::handle x) {
        return py::hasattr(x, "data_ptr");
    };
    auto is_int = [](py::handle x) {
        return py::isinstance<py::int_>(x);
    };

    py::list spec_key;
    for (auto arg : spec_args_) {
        if (is_tensor(arg)) {
            auto dtype = arg.attr("dtype");
            uintptr_t data_ptr = arg.attr("data_ptr")().cast<uintptr_t>();
            bool aligned = (data_ptr & (divisibility_ - 1)) == 0;
            spec_key.append(py::make_tuple(dtype, aligned));
        } else {
            spec_key.append(py::make_tuple(py::type::of(arg), arg));
        }
    }

    py::list dns_key;
    for (auto arg : dns_args_) {
        if (is_tensor(arg)) {
            dns_key.append(arg.attr("dtype"));
        } else if (!is_int(arg)) {
            dns_key.append(py::type::of(arg));
        } else {
            int64_t val = arg.cast<int64_t>();
            if (val >= -0x80000000LL && val <= 0x7FFFFFFFLL) {
                dns_key.append(py::str("i32"));
            } else if (val >= 0 && val <= 0xFFFFFFFFFFFFFFFFLL) {
                dns_key.append(py::str("u64"));
            } else {
                dns_key.append(py::str("i64"));
            }
        }
    }

    py::list result;
    auto list_append = [&](const py::list& src) {
        for (auto handle : src) {
            result.append(handle);
        }
    };
    list_append(spec_key);
    list_append(dns_key);
    list_append(const_args_);
    return result;
}

py::list libentry::ArgProcessor::getKArgs()
{
    return k_args_;
}