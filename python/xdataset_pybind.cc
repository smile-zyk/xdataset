// =============================================================================
// xdataset_pybind.cc — pybind11 bindings for the xdataset library (numpy-aligned)
//
// Core exported types (only these three data types are the public API):
//   Measurement  — scalar | vector | matrix value + unit (numpy-scalar-like)
//   DataArray    — a variable: self data + independent coordinates
//   Value        — unified Measurement-or-DataArray
//
// Supporting exports (needed to *use* the core types):
//   Unit, DataKind/DataType/DataArrayKind enums, and MultiIndexSelector
//   (used by DataArray.at / DataArray.select).
//
// No Dataset container and no HDF5/Touchstone IO is exported — this module
// is purely the in-memory value types.
//
// numpy alignment:
//   - .shape is a tuple, .ndim an int, .dtype a numpy dtype
//   - .values / .value expose the concrete data (numpy arrays for 1-D+)
//   - __array__ protocol: np.asarray(x) works
//   - indexing / slicing follows numpy semantics
//   - arithmetic mixes with Python scalars, numpy scalars and numpy arrays;
//     comparisons return numpy bool (scalars return Python bool)
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <boost/variant.hpp>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_array.h"
#include "measurement.h"
#include "multi_index_selector.h"
#include "unit.h"
#include "value.h"
#include "xdataset_predefine.h"

namespace py = pybind11;
using namespace xdataset;

// =============================================================================
// Conversion helpers
// =============================================================================

namespace {

// ---- numpy dtype for a DataType -------------------------------------------

py::dtype dtype_of(DataType t)
{
    switch (t)
    {
        case DataType::kReal:    return py::dtype::of<double>();
        case DataType::kInteger: return py::dtype::of<int>();
        case DataType::kComplex: return py::dtype::of<std::complex<double>>();
        case DataType::kBoolean: return py::dtype::of<bool>();
        case DataType::kString:  return py::dtype("O");  // object
    }
    return py::dtype::of<double>();
}

py::tuple shape_tuple(const DataShape& s)
{
    py::tuple t(s.dims.size());
    for (size_t i = 0; i < s.dims.size(); ++i)
        t[i] = py::int_(s.dims[i]);
    return t;
}

// ---- string tensors <-> nested Python lists -------------------------------

py::list vecxs_to_py(const VecXs& v)
{
    py::list out;
    for (Index i = 0; i < v.size(); ++i)
        out.append(py::str(v(i)));
    return out;
}

py::list matxs_to_py(const MatXs& m)
{
    py::list out;
    for (Index r = 0; r < m.dimension(0); ++r)
    {
        py::list row;
        for (Index c = 0; c < m.dimension(1); ++c)
            row.append(py::str(m(r, c)));
        out.append(row);
    }
    return out;
}

void fill_str_1d(py::handle h, std::vector<std::string>& out)
{
    for (py::handle item : h)
        out.push_back(py::cast<std::string>(item));
}

void fill_str_2d(py::handle h, std::vector<std::vector<std::string>>& out)
{
    for (py::handle row : h)
    {
        std::vector<std::string> r;
        for (py::handle item : row)
            r.push_back(py::cast<std::string>(item));
        out.push_back(std::move(r));
    }
}

void fill_str_3d(py::handle h, std::vector<std::vector<std::vector<std::string>>>& out)
{
    for (py::handle mat : h)
    {
        std::vector<std::vector<std::string>> m;
        for (py::handle row : mat)
        {
            std::vector<std::string> r;
            for (py::handle item : row)
                r.push_back(py::cast<std::string>(item));
            m.push_back(std::move(r));
        }
        out.push_back(std::move(m));
    }
}

int str_depth(py::handle h)
{
    if (py::isinstance<py::str>(h) || !py::isinstance<py::sequence>(h))
        return 0;
    if (py::len(h) == 0)
        return 1;
    return 1 + str_depth(py::reinterpret_borrow<py::sequence>(h)[0]);
}

// ---- Measurement -> Python value / numpy array ----------------------------

py::object measurement_value(const Measurement& m)
{
    const Measurement::Storage& s = m.storage();
    if (const double* p = boost::get<double>(&s))
        return py::float_(*p);
    if (const int* p = boost::get<int>(&s))
        return py::int_(*p);
    if (const std::complex<double>* p = boost::get<std::complex<double>>(&s))
        return py::cast(*p);
    if (const std::string* p = boost::get<std::string>(&s))
        return py::str(*p);
    if (const bool* p = boost::get<bool>(&s))
        return py::bool_(*p);
    if (const VecXd* p = boost::get<VecXd>(&s))
        return py::cast(*p);
    if (const VecXi* p = boost::get<VecXi>(&s))
        return py::cast(*p);
    if (const VecXcd* p = boost::get<VecXcd>(&s))
        return py::cast(*p);
    if (const VecXs* p = boost::get<VecXs>(&s))
        return py::array::ensure(vecxs_to_py(*p));
    if (const MatXd* p = boost::get<MatXd>(&s))
        return py::cast(*p);
    if (const MatXi* p = boost::get<MatXi>(&s))
        return py::cast(*p);
    if (const MatXcd* p = boost::get<MatXcd>(&s))
        return py::cast(*p);
    if (const MatXs* p = boost::get<MatXs>(&s))
        return py::array::ensure(matxs_to_py(*p));
    return py::none();
}

/// Always return a numpy array (0-d for scalars, 1-D/2-D for vector/matrix).
py::array measurement_to_numpy(const Measurement& m)
{
    py::object v = measurement_value(m);
    if (py::isinstance<py::array>(v))
        return v;
    return py::array::ensure(v);
}

// ---- DataSeries -> numpy (string data -> object ndarray) ------------------

py::array series_to_numpy(const DataSeries& s)
{
    const std::size_t n = s.size();
    const DataType dt = s.data_type();
    const DataKind k = s.data_kind();

    if (dt == DataType::kString)
    {
        py::list rows;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Measurement m = s.measurement_at(static_cast<Index>(i));
            if (k == DataKind::kScalar)
                rows.append(py::str(boost::get<std::string>(m.storage())));
            else if (k == DataKind::kVector)
                rows.append(vecxs_to_py(boost::get<VecXs>(m.storage())));
            else
                rows.append(matxs_to_py(boost::get<MatXs>(m.storage())));
        }
        py::object obj_dtype = py::dtype("O");
        return py::module_::import("numpy").attr("array")(rows, py::arg("dtype") = obj_dtype);
    }

    if (k == DataKind::kScalar)
    {
        if (dt == DataType::kReal)
        {
            py::array_t<double> a(n);
            double* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
                p[i] = s.scalar_at<double>(static_cast<Index>(i));
            return a;
        }
        if (dt == DataType::kInteger)
        {
            py::array_t<int> a(n);
            int* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
                p[i] = s.scalar_at<int>(static_cast<Index>(i));
            return a;
        }
        if (dt == DataType::kComplex)
        {
            py::array_t<std::complex<double>> a(n);
            auto* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
                p[i] = s.scalar_at<std::complex<double>>(static_cast<Index>(i));
            return a;
        }
        if (dt == DataType::kBoolean)
        {
            py::array_t<bool> a(n);
            bool* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
                p[i] = s.measurement_at(static_cast<Index>(i)).has_value();
            return a;
        }
    }
    else if (k == DataKind::kVector)
    {
        const Index w = s.data_shape().dims[0];
        if (dt == DataType::kReal)
        {
            py::array_t<double> a({n, static_cast<std::size_t>(w)});
            double* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
            {
                const VecXd& v = s.vector_at<double>(static_cast<Index>(i));
                for (Index j = 0; j < w; ++j)
                    p[i * static_cast<std::size_t>(w) + static_cast<std::size_t>(j)] = v(j);
            }
            return a;
        }
        if (dt == DataType::kInteger)
        {
            py::array_t<int> a({n, static_cast<std::size_t>(w)});
            int* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
            {
                const VecXi& v = s.vector_at<int>(static_cast<Index>(i));
                for (Index j = 0; j < w; ++j)
                    p[i * static_cast<std::size_t>(w) + static_cast<std::size_t>(j)] = v(j);
            }
            return a;
        }
        if (dt == DataType::kComplex)
        {
            py::array_t<std::complex<double>> a({n, static_cast<std::size_t>(w)});
            auto* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
            {
                const VecXcd& v = s.vector_at<std::complex<double>>(static_cast<Index>(i));
                for (Index j = 0; j < w; ++j)
                    p[i * static_cast<std::size_t>(w) + static_cast<std::size_t>(j)] = v(j);
            }
            return a;
        }
    }
    else if (k == DataKind::kMatrix)
    {
        const Index r = s.data_shape().dims[0];
        const Index c = s.data_shape().dims[1];
        if (dt == DataType::kReal)
        {
            py::array_t<double> a({n, static_cast<std::size_t>(r), static_cast<std::size_t>(c)});
            double* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
            {
                const MatXd& m = s.matrix_at<double>(static_cast<Index>(i));
                for (Index ri = 0; ri < r; ++ri)
                    for (Index ci = 0; ci < c; ++ci)
                        p[(i * static_cast<std::size_t>(r) + static_cast<std::size_t>(ri)) *
                              static_cast<std::size_t>(c) + static_cast<std::size_t>(ci)] = m(ri, ci);
            }
            return a;
        }
        if (dt == DataType::kInteger)
        {
            py::array_t<int> a({n, static_cast<std::size_t>(r), static_cast<std::size_t>(c)});
            int* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
            {
                const MatXi& m = s.matrix_at<int>(static_cast<Index>(i));
                for (Index ri = 0; ri < r; ++ri)
                    for (Index ci = 0; ci < c; ++ci)
                        p[(i * static_cast<std::size_t>(r) + static_cast<std::size_t>(ri)) *
                              static_cast<std::size_t>(c) + static_cast<std::size_t>(ci)] = m(ri, ci);
            }
            return a;
        }
        if (dt == DataType::kComplex)
        {
            py::array_t<std::complex<double>> a({n, static_cast<std::size_t>(r), static_cast<std::size_t>(c)});
            auto* p = a.mutable_data();
            for (std::size_t i = 0; i < n; ++i)
            {
                const MatXcd& m = s.matrix_at<std::complex<double>>(static_cast<Index>(i));
                for (Index ri = 0; ri < r; ++ri)
                    for (Index ci = 0; ci < c; ++ci)
                        p[(i * static_cast<std::size_t>(r) + static_cast<std::size_t>(ri)) *
                              static_cast<std::size_t>(c) + static_cast<std::size_t>(ci)] = m(ri, ci);
            }
            return a;
        }
    }

    // Fallback: list of Measurement values as object array.
    py::list l;
    for (std::size_t i = 0; i < n; ++i)
        l.append(measurement_value(s.measurement_at(static_cast<Index>(i))));
    return py::module_::import("numpy").attr("array")(l,
        py::arg("dtype") = py::dtype("O"));
}

// ---- Python object -> Measurement (auto-detects scalar / vector / matrix) --

Measurement measurement_from_python(const py::object& obj, const Unit& unit)
{
    py::array arr = py::array::ensure(obj);
    if (arr)
    {
        py::array arr_c = py::array::ensure(arr, py::array::c_style);
        const char kind = arr.dtype().kind();
        if (kind == 'b')
        {
            py::array_t<bool> a = py::array_t<bool>::ensure(arr_c);
            if (!a || a.ndim() != 0)
                throw py::type_error("boolean Measurement must be a scalar");
            Measurement m = Measurement::Boolean(*a.data());
            if (unit.has_dimension())
                throw py::value_error("Boolean measurements cannot have a unit");
            return m;
        }
        if (kind == 'f')
        {
            py::array_t<double> a = py::array_t<double>::ensure(arr_c);
            if (!a) throw py::type_error("cannot convert array to float64");
            auto buf = a.request();
            const double* data = static_cast<const double*>(buf.ptr);
            Measurement m;
            if (buf.ndim == 0)
                m = Measurement::Real(data[0]);
            else if (buf.ndim == 1)
                m = Measurement::Vector(VecXd(Eigen::Map<const VecXd>(data, buf.shape[0])));
            else if (buf.ndim == 2)
                m = Measurement::Matrix(MatXd(Eigen::Map<const MatXd>(data, buf.shape[0], buf.shape[1])));
            else
                throw py::value_error("Measurement must be scalar, 1-D or 2-D");
            m.set_unit(unit);
            return m;
        }
        if (kind == 'c')
        {
            py::array_t<std::complex<double>> a = py::array_t<std::complex<double>>::ensure(arr_c);
            if (!a) throw py::type_error("cannot convert array to complex128");
            auto buf = a.request();
            const std::complex<double>* data = static_cast<const std::complex<double>*>(buf.ptr);
            Measurement m;
            if (buf.ndim == 0)
                m = Measurement::Complex(data[0]);
            else if (buf.ndim == 1)
                m = Measurement::Vector(VecXcd(Eigen::Map<const VecXcd>(data, buf.shape[0])));
            else if (buf.ndim == 2)
                m = Measurement::Matrix(MatXcd(Eigen::Map<const MatXcd>(data, buf.shape[0], buf.shape[1])));
            else
                throw py::value_error("Measurement must be scalar, 1-D or 2-D");
            m.set_unit(unit);
            return m;
        }
        if (kind == 'i' || kind == 'u')
        {
            py::array_t<int> a = py::array_t<int>::ensure(arr_c);
            if (!a) throw py::type_error("cannot convert array to int32");
            auto buf = a.request();
            const int* data = static_cast<const int*>(buf.ptr);
            Measurement m;
            if (buf.ndim == 0)
                m = Measurement::Integer(data[0]);
            else if (buf.ndim == 1)
                m = Measurement::Vector(VecXi(Eigen::Map<const VecXi>(data, buf.shape[0])));
            else if (buf.ndim == 2)
                m = Measurement::Matrix(MatXi(Eigen::Map<const MatXi>(data, buf.shape[0], buf.shape[1])));
            else
                throw py::value_error("Measurement must be scalar, 1-D or 2-D");
            m.set_unit(unit);
            return m;
        }
        if (kind == 'U' || kind == 'S')
        {
            if (arr.ndim() == 0)
            {
                Measurement m = Measurement::String(py::cast<std::string>(py::str(arr.attr("item")())));
                m.set_unit(unit);
                return m;
            }
            if (arr.ndim() == 1)
            {
                std::vector<std::string> v1;
                for (py::handle item : arr)
                    v1.push_back(py::cast<std::string>(py::str(item)));
                VecXs out(static_cast<Index>(v1.size()));
                for (Index i = 0; i < static_cast<Index>(v1.size()); ++i)
                    out(i) = v1[i];
                Measurement m = Measurement::Vector(out);
                m.set_unit(unit);
                return m;
            }
            throw py::value_error("string Measurement must be scalar or 1-D");
        }
        throw py::type_error("unsupported numpy dtype for Measurement");
    }

    // Non-array Python values.
    if (py::isinstance<py::bool_>(obj))
        return Measurement::Boolean(obj.cast<bool>());
    if (py::isinstance<py::int_>(obj))
        return Measurement::Integer(obj.cast<int>());
    if (py::isinstance<py::float_>(obj))
        return Measurement::Real(obj.cast<double>());
    if (PyComplex_Check(obj.ptr()))
        return Measurement::Complex(py::cast<std::complex<double>>(obj));
    if (py::isinstance<py::str>(obj))
        return Measurement::String(obj.cast<std::string>());
    throw py::type_error("cannot convert object to Measurement");
}

// ---- Python object -> DataSeries (used by DataArray.from_numpy) -----------

template <typename T>
DataSeries series_from_numeric(const py::array& arr, const Unit& unit)
{
    py::array_t<T> a = py::array_t<T>::ensure(arr);
    if (!a)
        throw std::invalid_argument("cannot convert array to the requested numeric dtype");
    auto buf = a.request();
    const T* data = static_cast<const T*>(buf.ptr);

    if (buf.ndim == 1)
    {
        return DataSeries::CreateScalarFromVector<T>(
            std::vector<T>(data, data + buf.shape[0]), unit);
    }
    if (buf.ndim == 2)
    {
        std::vector<std::vector<T>> rows(buf.shape[0]);
        for (py::ssize_t i = 0; i < buf.shape[0]; ++i)
        {
            const T* row = data + i * buf.shape[1];
            rows[i].assign(row, row + buf.shape[1]);
        }
        return DataSeries::CreateVectorFromNestedVector<T>(rows, unit);
    }
    if (buf.ndim == 3)
    {
        std::vector<std::vector<std::vector<T>>> rows(buf.shape[0]);
        const py::ssize_t r = buf.shape[1];
        const py::ssize_t c = buf.shape[2];
        for (py::ssize_t i = 0; i < buf.shape[0]; ++i)
        {
            rows[i].resize(r);
            for (py::ssize_t j = 0; j < r; ++j)
            {
                const T* row = data + (i * r + j) * c;
                rows[i][j].assign(row, row + c);
            }
        }
        return DataSeries::CreateMatrixFromNestedVector<T>(rows, unit);
    }
    throw std::invalid_argument(
        "DataSeries input must be 1-D (scalar cells), 2-D (vector cells) or 3-D (matrix cells)");
}

DataSeries series_from_strings(const py::object& obj, const Unit& unit)
{
    py::list l = py::cast<py::list>(obj);
    const int depth = str_depth(l);
    if (depth == 1)
    {
        std::vector<std::string> v1;
        fill_str_1d(l, v1);
        return DataSeries::CreateScalarFromVector(v1, unit);
    }
    if (depth == 2)
    {
        std::vector<std::vector<std::string>> v2;
        fill_str_2d(l, v2);
        return DataSeries::CreateVectorFromNestedVector(v2, unit);
    }
    if (depth >= 3)
    {
        std::vector<std::vector<std::vector<std::string>>> v3;
        fill_str_3d(l, v3);
        return DataSeries::CreateMatrixFromNestedVector(v3, unit);
    }
    return DataSeries::CreateScalarFromVector(std::vector<std::string>{}, unit);
}

DataSeries series_from_python(const py::object& obj, const Unit& unit)
{
    py::array arr = py::array::ensure(obj);
    if (arr)
    {
        py::array arr_c = py::array::ensure(arr, py::array::c_style);
        const char kind = arr.dtype().kind();
        if (kind == 'c')
            return series_from_numeric<std::complex<double>>(arr_c, unit);
        if (kind == 'f')
            return series_from_numeric<double>(arr_c, unit);
        if (kind == 'i' || kind == 'u')
            return series_from_numeric<int>(arr_c, unit);
        if (kind == 'U' || kind == 'S')
            return series_from_strings(obj, unit);
    }
    return series_from_strings(obj, unit);
}

// ---- operand coercion for operators --------------------------------------

/// Coerce a Python operand to a Measurement (scalars, numpy arrays, lists).
Measurement to_measurement(const py::object& obj)
{
    if (py::isinstance<Measurement>(obj))
        return obj.cast<Measurement>();
    return measurement_from_python(obj, Unit());
}

/// Coerce a Python operand to a Value.
Value to_value(const py::object& obj)
{
    if (py::isinstance<Value>(obj))
        return obj.cast<Value>();
    if (py::isinstance<Measurement>(obj))
        return Value(obj.cast<Measurement>());
    if (py::isinstance<DataArray>(obj))
        return Value(obj.cast<DataArray>());
    if (py::isinstance<py::bool_>(obj))
        return Value::Boolean(obj.cast<bool>());
    if (py::isinstance<py::int_>(obj))
        return Value::Integer(obj.cast<int>());
    if (py::isinstance<py::float_>(obj))
        return Value::Real(obj.cast<double>());
    if (PyComplex_Check(obj.ptr()))
        return Value::Complex(py::cast<std::complex<double>>(obj));
    if (py::isinstance<py::str>(obj))
        return Value::String(obj.cast<std::string>());
    return Value(measurement_from_python(obj, Unit()));
}

/// Apply an optional dtype request (the dtype argument of __array__).
py::object apply_dtype(const py::array& arr, py::handle dtype)
{
    if (dtype.is_none())
        return arr;
    return arr.attr("astype")(dtype);
}

// numpy scalar truth helper
bool measurement_truth(const Measurement& m)
{
    const auto& s = m.storage();
    if (const double* p = boost::get<double>(&s)) return *p != 0.0;
    if (const int* p = boost::get<int>(&s)) return *p != 0;
    if (const std::complex<double>* p = boost::get<std::complex<double>>(&s)) return *p != 0.0;
    if (const bool* p = boost::get<bool>(&s)) return *p;
    throw py::type_error("cannot determine the truth value");
}

}  // namespace

// =============================================================================
// Module definition
// =============================================================================

PYBIND11_MODULE(xdataset, m)
{
    m.doc() = "xdataset Python bindings: Measurement, DataArray, Value "
              "(numpy-aligned).";

    // -----------------------------------------------------------------------
    // Enums
    // -----------------------------------------------------------------------
    py::enum_<DataKind>(m, "DataKind")
        .value("kScalar", DataKind::kScalar)
        .value("kVector", DataKind::kVector)
        .value("kMatrix", DataKind::kMatrix);

    py::enum_<DataType>(m, "DataType")
        .value("kReal", DataType::kReal)
        .value("kInteger", DataType::kInteger)
        .value("kComplex", DataType::kComplex)
        .value("kString", DataType::kString)
        .value("kBoolean", DataType::kBoolean);

    py::enum_<DataArrayKind>(m, "DataArrayKind")
        .value("kDependent", DataArrayKind::kDependent)
        .value("kIndependent", DataArrayKind::kIndependent);

    // -----------------------------------------------------------------------
    // MultiIndexSelector (support type — DataArray.at / DataArray.select)
    // -----------------------------------------------------------------------
    py::class_<MultiIndexSelector>(m, "MultiIndexSelector",
        "Selector for DataArray.at / DataArray.select.")
        .def_static("Any", &MultiIndexSelector::Any)
        .def_static("Equal", &MultiIndexSelector::Equal, py::arg("idx"))
        .def_static("In", &MultiIndexSelector::In, py::arg("indices"))
        .def_property_readonly("kind", &MultiIndexSelector::kind)
        .def("is_any", &MultiIndexSelector::is_any)
        .def("is_equal", &MultiIndexSelector::is_equal)
        .def("is_in", &MultiIndexSelector::is_in)
        .def_property_readonly("equal_value", &MultiIndexSelector::equal_value)
        .def_property_readonly("in_values", &MultiIndexSelector::in_values)
        .def("resolve", &MultiIndexSelector::resolve, py::arg("width"))
        .def("matches", &MultiIndexSelector::matches, py::arg("idx"))
        .def("__repr__", [](const MultiIndexSelector& s) {
            if (s.is_any()) return std::string("MultiIndexSelector.Any()");
            if (s.is_equal()) return "MultiIndexSelector.Equal(" + std::to_string(s.equal_value()) + ")";
            std::string out = "MultiIndexSelector.In([";
            for (size_t i = 0; i < s.in_values().size(); ++i)
            {
                if (i) out += ", ";
                out += std::to_string(s.in_values()[i]);
            }
            out += "])";
            return out;
        });

    // -----------------------------------------------------------------------
    // Unit (support type — used by .unit attributes)
    // -----------------------------------------------------------------------
    py::class_<Unit>(m, "Unit",
        "A physical unit. Unit.parse('GHz') accepts the REL vocabulary.")
        .def(py::init<>())
        .def_static("parse", &Unit::parse, py::arg("s"))
        .def_static("none", &Unit::None, "The dimensionless unit.")
        .def("to_string", &Unit::to_string)
        .def("__str__", &Unit::to_string)
        .def("__repr__", [](const Unit& u) { return "Unit('" + u.to_string() + "')"; })
        .def_property_readonly("multiplier", &Unit::multiplier)
        .def_property_readonly("is_affine", &Unit::is_affine)
        .def_property_readonly("is_canonical", &Unit::is_canonical)
        .def_property_readonly("is_dimensionless", &Unit::is_dimensionless)
        .def("has_dimension", &Unit::has_dimension)
        .def("same_dimension", &Unit::same_dimension, py::arg("other"))
        .def("canonicalized", &Unit::canonicalized)
        .def("best_display", [](const Unit& u, double value) {
            const UnitScale bs = u.best_display(value);
            return py::make_tuple(bs.scale, bs.name);
        }, py::arg("value"))
        .def("__eq__", &Unit::operator==)
        .def("__ne__", &Unit::operator!=);

    // -----------------------------------------------------------------------
    // Measurement — numpy-scalar-like
    // -----------------------------------------------------------------------
    py::class_<Measurement>(m, "Measurement",
        "A scalar / vector / matrix value carrying a Unit. numpy-aligned: "
        ".shape, .ndim, .dtype, .values, np.asarray(m), indexing, operators.")
        .def(py::init<>(), "Default: Real 0, dimensionless.")
        .def_static("from_numpy", [](const py::object& data, const Unit& unit) {
            return measurement_from_python(data, unit);
        }, py::arg("data"), py::arg("unit") = Unit(),
            "Build a Measurement from a scalar, numpy array, or nested list.\n"
            "0-d -> scalar, 1-D -> vector cell, 2-D -> matrix cell.")
        .def_static("Real", &Measurement::Real, py::arg("value"))
        .def_static("Integer", &Measurement::Integer, py::arg("value"))
        .def_static("Complex", &Measurement::Complex, py::arg("value"))
        .def_static("String", &Measurement::String, py::arg("value"))
        .def_static("Boolean", &Measurement::Boolean, py::arg("value"))
        .def_static("Vector",
            static_cast<Measurement(*)(const VecXd&)>(&Measurement::Vector), py::arg("v"))
        .def_static("Vector",
            static_cast<Measurement(*)(const VecXi&)>(&Measurement::Vector), py::arg("v"))
        .def_static("Vector",
            static_cast<Measurement(*)(const VecXcd&)>(&Measurement::Vector), py::arg("v"))
        .def_static("Vector", [](const py::list& v) {
            VecXs out(static_cast<Index>(v.size()));
            for (Index i = 0; i < static_cast<Index>(v.size()); ++i)
                out(i) = v[i].cast<std::string>();
            return Measurement::Vector(out);
        }, py::arg("strings"))
        .def_static("Matrix",
            static_cast<Measurement(*)(const MatXd&)>(&Measurement::Matrix), py::arg("m"))
        .def_static("Matrix",
            static_cast<Measurement(*)(const MatXi&)>(&Measurement::Matrix), py::arg("m"))
        .def_static("Matrix",
            static_cast<Measurement(*)(const MatXcd&)>(&Measurement::Matrix), py::arg("m"))
        .def_static("Matrix", [](const py::list& rows) {
            const Index nr = static_cast<Index>(rows.size());
            const Index nc = nr > 0 ? static_cast<Index>(py::len(rows[0])) : 0;
            MatXs out(nr, nc);
            for (Index r = 0; r < nr; ++r)
                for (Index c = 0; c < nc; ++c)
                    out(r, c) = rows[r].cast<py::list>()[c].cast<std::string>();
            return Measurement::Matrix(out);
        }, py::arg("strings"))

        // numpy-aligned metadata
        .def_property_readonly("shape", [](const Measurement& m) {
            return shape_tuple(m.shape());
        })
        .def_property_readonly("ndim", [](const Measurement& m) {
            return static_cast<int>(m.shape().size());
        })
        .def_property_readonly("dtype", [](const Measurement& m) {
            return dtype_of(m.data_type());
        })
        .def_property_readonly("size", [](const Measurement& m) {
            return static_cast<py::ssize_t>(m.element_count());
        })
        .def_property_readonly("kind", &Measurement::data_kind)
        .def_property("unit",
            [](const Measurement& m) -> const Unit& { return m.unit(); },
            [](Measurement& m, const Unit& u) { m.set_unit(u); },
            py::return_value_policy::reference_internal)

        // data access
        .def_property_readonly("value", [](const Measurement& m) {
            return measurement_value(m);
        }, "The concrete value: Python scalar or numpy array.")
        .def_property_readonly("values", [](const Measurement& m) {
            return measurement_to_numpy(m);
        }, "Always a numpy array (0-d for scalars).")
        .def("tolist", [](const Measurement& m) {
            return measurement_to_numpy(m).attr("tolist")();
        })
        .def("has_value", &Measurement::has_value)

        // numpy protocols
        .def("__array__", [](const Measurement& m, py::handle dtype, py::handle copy) {
            return apply_dtype(measurement_to_numpy(m), dtype);
        }, py::arg("dtype") = py::none(), py::arg("copy") = py::none())
        .def("__float__", [](const Measurement& m) {
            if (m.data_kind() != DataKind::kScalar)
                throw py::type_error("only a scalar Measurement can be converted to float");
            const auto& s = m.storage();
            if (const double* p = boost::get<double>(&s)) return *p;
            if (const int* p = boost::get<int>(&s)) return static_cast<double>(*p);
            if (const std::complex<double>* p = boost::get<std::complex<double>>(&s)) return p->real();
            if (const bool* p = boost::get<bool>(&s)) return *p ? 1.0 : 0.0;
            throw py::type_error("cannot convert to float");
        })
        .def("__int__", [](const Measurement& m) {
            if (m.data_kind() != DataKind::kScalar)
                throw py::type_error("only a scalar Measurement can be converted to int");
            const auto& s = m.storage();
            if (const int* p = boost::get<int>(&s)) return *p;
            if (const double* p = boost::get<double>(&s)) return static_cast<int>(*p);
            if (const bool* p = boost::get<bool>(&s)) return *p ? 1 : 0;
            throw py::type_error("cannot convert to int");
        })
        .def("__complex__", [](const Measurement& m) {
            if (m.data_kind() != DataKind::kScalar)
                throw py::type_error("only a scalar Measurement can be converted to complex");
            const auto& s = m.storage();
            if (const std::complex<double>* p = boost::get<std::complex<double>>(&s)) return *p;
            if (const double* p = boost::get<double>(&s)) return std::complex<double>(*p);
            if (const int* p = boost::get<int>(&s)) return std::complex<double>(*p);
            throw py::type_error("cannot convert to complex");
        })
        .def("__bool__", [](const Measurement& m) {
            if (m.data_kind() != DataKind::kScalar)
                throw py::type_error("the truth value of a non-scalar Measurement is ambiguous");
            return measurement_truth(m);
        })
        .def("__len__", [](const Measurement& m) {
            if (m.data_kind() == DataKind::kScalar)
                throw py::type_error("len() of unsized object");
            return static_cast<py::ssize_t>(m.shape()[0]);
        })
        .def("__iter__", [](const Measurement& m) {
            if (m.data_kind() == DataKind::kScalar)
                throw py::type_error("iteration over a 0-d Measurement");
            py::array arr = measurement_to_numpy(m);
            py::list out;
            for (py::ssize_t i = 0; i < arr.shape(0); ++i)
                out.append(arr.attr("__getitem__")(i));
            return py::iter(out);
        })

        // indexing: int -> Measurement (keeps unit); slice/tuple -> ndarray
        .def("__getitem__", [](const Measurement& m, const py::object& key) -> py::object {
            py::array arr = measurement_to_numpy(m);
            if (py::isinstance<py::int_>(key) && arr.ndim() >= 1)
            {
                py::object sub = arr.attr("__getitem__")(key);
                return py::cast(measurement_from_python(sub, m.unit()));
            }
            return arr.attr("__getitem__")(key);
        })

        // element access
        .def("element_at",
            static_cast<Measurement(Measurement::*)(Index) const>(&Measurement::element_at),
            py::arg("i"))
        .def("element_at",
            static_cast<Measurement(Measurement::*)(Index, Index) const>(&Measurement::element_at),
            py::arg("r"), py::arg("c"))

        // arithmetic (numpy-aligned mixing with scalars / arrays)
        .def("__add__",  [](const Measurement& a, const py::object& b) { return a + to_measurement(b); })
        .def("__radd__", [](const Measurement& a, const py::object& b) { return to_measurement(b) + a; })
        .def("__sub__",  [](const Measurement& a, const py::object& b) { return a - to_measurement(b); })
        .def("__rsub__", [](const Measurement& a, const py::object& b) { return to_measurement(b) - a; })
        .def("__mul__",  [](const Measurement& a, const py::object& b) { return a * to_measurement(b); })
        .def("__rmul__", [](const Measurement& a, const py::object& b) { return to_measurement(b) * a; })
        .def("__truediv__",  [](const Measurement& a, const py::object& b) { return a / to_measurement(b); })
        .def("__rtruediv__", [](const Measurement& a, const py::object& b) { return to_measurement(b) / a; })
        .def("__mod__",  [](const Measurement& a, const py::object& b) { return a % to_measurement(b); })
        .def("__rmod__", [](const Measurement& a, const py::object& b) { return to_measurement(b) % a; })
        .def("__pow__",  [](const Measurement& a, const py::object& b) {
            return xdataset::pow(Value(a), Value(to_measurement(b))).as_measurement();
        })
        .def("__rpow__", [](const Measurement& a, const py::object& b) {
            return xdataset::pow(Value(to_measurement(b)), Value(a)).as_measurement();
        })
        .def("__neg__", [](const Measurement& a) { return -a; })

        // comparisons -> Python bool for scalars, numpy bool array otherwise
        .def("__eq__", [](const Measurement& a, const py::object& b) -> py::object {
            const Measurement r = a == to_measurement(b);
            if (r.data_kind() == DataKind::kScalar && r.data_type() == DataType::kBoolean)
                return py::bool_(boost::get<bool>(r.storage()));
            return measurement_to_numpy(r).attr("astype")("bool");
        })
        .def("__ne__", [](const Measurement& a, const py::object& b) -> py::object {
            const Measurement r = a != to_measurement(b);
            if (r.data_kind() == DataKind::kScalar && r.data_type() == DataType::kBoolean)
                return py::bool_(boost::get<bool>(r.storage()));
            return measurement_to_numpy(r).attr("astype")("bool");
        })
        .def("__lt__", [](const Measurement& a, const py::object& b) -> py::object {
            const Measurement r = a < to_measurement(b);
            if (r.data_kind() == DataKind::kScalar && r.data_type() == DataType::kBoolean)
                return py::bool_(boost::get<bool>(r.storage()));
            return measurement_to_numpy(r).attr("astype")("bool");
        })
        .def("__gt__", [](const Measurement& a, const py::object& b) -> py::object {
            const Measurement r = a > to_measurement(b);
            if (r.data_kind() == DataKind::kScalar && r.data_type() == DataType::kBoolean)
                return py::bool_(boost::get<bool>(r.storage()));
            return measurement_to_numpy(r).attr("astype")("bool");
        })
        .def("__le__", [](const Measurement& a, const py::object& b) -> py::object {
            const Measurement r = a <= to_measurement(b);
            if (r.data_kind() == DataKind::kScalar && r.data_type() == DataType::kBoolean)
                return py::bool_(boost::get<bool>(r.storage()));
            return measurement_to_numpy(r).attr("astype")("bool");
        })
        .def("__ge__", [](const Measurement& a, const py::object& b) -> py::object {
            const Measurement r = a >= to_measurement(b);
            if (r.data_kind() == DataKind::kScalar && r.data_type() == DataType::kBoolean)
                return py::bool_(boost::get<bool>(r.storage()));
            return measurement_to_numpy(r).attr("astype")("bool");
        })

        // formatting
        .def("to_string", &Measurement::to_string)
        .def("__str__", &Measurement::to_string)
        .def("__repr__", [](const Measurement& m) {
            return "Measurement(" + m.to_string() + ")";
        })
        .def("canonicalized", &Measurement::canonicalized)
        .def_property_readonly("is_canonicalized", &Measurement::is_canonicalized);

    // -----------------------------------------------------------------------
    // DataArray — numpy/xarray-aligned array type
    // -----------------------------------------------------------------------
    py::class_<DataArray>(m, "DataArray",
        "A variable with data (and optionally independent coordinates). "
        "numpy-aligned: .values, .shape, .ndim, .dtype, .unit, np.asarray(da), "
        "indexing/slicing, operators (return DataArray), comparisons "
        "(return numpy bool arrays).")
        .def_static("from_numpy", [](const py::object& data, const Unit& unit) {
            return DataArray::CreateIndependent(series_from_python(data, unit));
        }, py::arg("data"), py::arg("unit") = Unit(),
            "Build an independent DataArray from a numpy array or nested lists.\n"
            "1-D -> scalar cells, 2-D -> vector cells, 3-D -> matrix cells.")
        .def_static("create_dependent", [](const py::object& data,
                                            const py::dict& indep_variables,
                                            const Unit& unit) {
            tsl::ordered_map<std::string, const DataArray*> vars;
            for (const auto& kv : indep_variables)
            {
                const std::string name = py::cast<std::string>(kv.first);
                const DataArray* var = py::cast<DataArray*>(kv.second);
                if (!var)
                    throw py::type_error("create_dependent: indep values must be DataArray");
                vars[name] = var;
            }
            return DataArray::CreateDependent(series_from_python(data, unit), vars);
        }, py::arg("data"), py::arg("indep_variables"), py::arg("unit") = Unit(),
            "Build a Dependent DataArray from data (leaf-count rows) and a dict of\n"
            "{name: Independent DataArray} coordinates. Each independent's last\n"
            "dimension size forms the multi-dimension spec; the product must equal\n"
            "the number of data rows.")

        // numpy-aligned metadata
        .def_property_readonly("values", [](const DataArray& a) {
            return series_to_numpy(a.data());
        }, "The dependent data as a numpy array.")
        .def_property_readonly("shape", [](const DataArray& a) {
            return series_to_numpy(a.data()).attr("shape");
        })
        .def_property_readonly("ndim", [](const DataArray& a) {
            return series_to_numpy(a.data()).attr("ndim");
        })
        .def_property_readonly("dtype", [](const DataArray& a) {
            return dtype_of(a.data().data_type());
        })
        .def_property_readonly("size", [](const DataArray& a) {
            return static_cast<py::ssize_t>(a.data().size());
        })
        .def_property_readonly("kind", &DataArray::data_kind)
        .def_property_readonly("rank", [](const DataArray& a) {
            return static_cast<int>(a.multi_dimension_spec().rank());
        }, "Number of dimensions (independent variable count).")
        .def_property_readonly("dims", [](const DataArray& a) {
            py::list out;
            for (const auto& d : a.multi_dimension_spec().dims())
                out.append(static_cast<int>(d.element_count()));
            return out;
        }, "Size of each dimension (regular: the size; ragged: the group count).")
        .def_property_readonly("unit",
            [](const DataArray& a) -> const Unit& { return a.data().unit(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly("indep_names", &DataArray::indep_names)
        .def_property_readonly("coords", [](const DataArray& a) {
            py::dict d;
            for (const auto& name : a.indep_names())
                d[py::str(name)] = series_to_numpy(a.indep_data(name));
            return d;
        }, "Independent coordinates as {name: numpy array}.")

        // independent-variable access
        .def("indep",
            static_cast<DataArray(DataArray::*)(Index) const>(&DataArray::indep),
            py::arg("index"), "The index-th independent variable as a DataArray "
                              "(1-based, matching the C++ API).")
        .def("indep",
            static_cast<DataArray(DataArray::*)(const std::string&) const>(&DataArray::indep),
            py::arg("name"), "The named independent variable as a DataArray.")
        .def("indep_data", [](const DataArray& a, Index index) {
            return series_to_numpy(a.indep_data(index));
        }, py::arg("index"), "The index-th independent-variable data as a numpy array "
                             "(1-based, matching the C++ API).")
        .def("indep_data", [](const DataArray& a, const std::string& name) {
            return series_to_numpy(a.indep_data(name));
        }, py::arg("name"), "The named independent-variable data as a numpy array.")

        // numpy protocols
        .def("__array__", [](const DataArray& a, py::handle dtype, py::handle copy) {
            return apply_dtype(series_to_numpy(a.data()), dtype);
        }, py::arg("dtype") = py::none(), py::arg("copy") = py::none())
        .def("__len__", [](const DataArray& a) {
            return static_cast<py::ssize_t>(a.data().size());
        })
        .def("__iter__", [](const DataArray& a) {
            py::array arr = series_to_numpy(a.data());
            py::list out;
            for (py::ssize_t i = 0; i < arr.shape(0); ++i)
                out.append(arr.attr("__getitem__")(i));
            return out;
        })
        // numpy semantics: da[i] / da[a:b] -> ndarray of the dependent data
        .def("__getitem__", [](const DataArray& a, const py::object& key) {
            return series_to_numpy(a.data()).attr("__getitem__")(key);
        })
        .def("tolist", [](const DataArray& a) {
            return series_to_numpy(a.data()).attr("tolist")();
        })

        // row access keeps the unit
        .def("row", [](const DataArray& a, Index i) {
            return a.data().measurement_at(i);
        }, py::arg("i"), "The i-th row as a Measurement (keeps unit).")

        // with_self_data: keep everything, replace only the self (kSelf) data
        .def("with_self_data", [](const DataArray& a, const py::object& new_data,
                                  const py::object& unit_opt) {
            Unit unit = a.data().unit();
            if (!unit_opt.is_none())
                unit = unit_opt.cast<Unit>();
            DataSeries new_self = series_from_python(new_data, unit);

            // Row-count check: Dependent self holds one row per leaf;
            // Independent self holds the raw innermost-dimension rows.
            const std::size_t rank = a.multi_dimension_spec().rank();
            std::size_t expected = 0;
            if (a.data_kind() == DataArrayKind::kDependent)
                expected = a.multi_dimension_spec().compute_cell_count();
            else if (rank > 0)
                expected = a.multi_dimension_spec().dim(
                    static_cast<Index>(rank - 1)).element_count();
            if (new_self.size() != expected)
            {
                throw py::value_error(
                    "with_self_data: new data row count (" + std::to_string(new_self.size()) +
                    ") must match the current self data row count (" +
                    std::to_string(expected) + ")");
            }

            DataArrayCreateInfo info;
            info.kind = a.data_kind();
            info.multi_dimension_spec = a.multi_dimension_spec();
            auto it = a.datas().begin();
            std::size_t idx = 0;
            for (; it != a.datas().end(); ++it, ++idx)
            {
                if (idx == a.datas().size() - 1)        // kSelf entry is last
                    info.datas[DataArray::kSelf] = std::move(new_self);
                else
                    info.datas[it->first] = it->second;  // dims/coords untouched
            }
            return DataArray(std::move(info));
        }, py::arg("new_data"), py::arg("unit") = py::none(),
            "Return a new DataArray identical to this one except that the self "
            "(kSelf) data — the dependent values of a Dependent DataArray, or "
            "the raw innermost-dimension values of an Independent DataArray — "
            "is replaced by new_data. Dimensions, coordinates, kind and unit "
            "are preserved; the unit can be overridden with unit=. The row "
            "count of new_data must match the current self data.")

        // selection: integer index list -> new DataArray
        .def("isel", [](const DataArray& a, const std::vector<Index>& rows) {
            return a.select({MultiIndexSelector::In(rows)});
        }, py::arg("rows"), "Select rows by index list; returns a new DataArray.")
        .def("at", [](const DataArray& a, const std::vector<MultiIndexSelector>& sel) {
            return a.at(sel);
        }, py::arg("selectors"))
        .def("select", [](const DataArray& a, const std::vector<MultiIndexSelector>& sel) {
            return a.select(sel);
        }, py::arg("selectors"))

        // innermost-dimension reduction
        .def("min", &DataArray::min,
            "Reduce along the innermost dimension by taking the group-wise "
            "minimum. Lowers the rank by one; when no dimension remains the "
            "result is a single-value Independent DataArray. Scalar data only; "
            "complex compares by magnitude; strings lexicographically.")
        .def("max", &DataArray::max,
            "Reduce along the innermost dimension by taking the group-wise "
            "maximum. Lowers the rank by one; when no dimension remains the "
            "result is a single-value Independent DataArray. Scalar data only; "
            "complex compares by magnitude; strings lexicographically.")
        .def("reduce", [](const DataArray& a, py::function fn) {
            const std::size_t rank = a.multi_dimension_spec().rank();
            if (rank == 0)
                throw py::value_error("reduce requires at least one dimension");

            // Groups of kSelf row indices, one per outer prefix of the
            // innermost dimension.  For Independent data the innermost raw
            // rows are shared by every outer prefix, exactly like min()/max().
            std::vector<std::vector<Index>> group_rows;
            if (rank >= 2)
            {
                a.multi_dimension_spec().for_each_group_at_dim(
                    static_cast<Index>(rank - 2),
                    [&](const MultiDimensionSpec::DimGroup& g)
                    {
                        std::vector<Index> rows;
                        a.multi_dimension_spec().for_each_leaf_row(
                            [&](const MultiDimensionSpec::LeafRow& leaf)
                            {
                                rows.push_back(
                                    (a.data_kind() == DataArrayKind::kIndependent)
                                        ? leaf.dimension_row_indices[rank - 1]
                                        : leaf.row_flat);
                            },
                            g.flat_start, g.flat_end);
                        group_rows.push_back(std::move(rows));
                    });
            }
            else
            {
                std::vector<Index> rows;
                for (Index f = 0; f < static_cast<Index>(a.data().size()); ++f)
                    rows.push_back(f);
                group_rows.push_back(std::move(rows));
            }

            // fn(group_values: np.ndarray) -> scalar, applied per group.
            std::vector<Measurement> results;
            results.reserve(group_rows.size());
            for (const auto& rows : group_rows)
            {
                // DataSeries::at() only supports vector/matrix cells, so build
                // the group subset row-by-row (works for every cell kind).
                const DataSeries& src = a.data();
                DataSeries sub(src.data_kind(), src.data_type(), src.data_shape());
                for (Index r : rows)
                    sub.append(src.measurement_at(r));
                const py::object res = fn(series_to_numpy(sub));
                results.push_back(measurement_from_python(res, a.data().unit()));
            }

            // Result series takes its dtype/shape from the first result so a
            // fn like np.mean can promote int data to float.
            if (results.empty())
                throw py::value_error("reduce produced no groups");
            DataSeries out(results.front().data_kind(), results.front().data_type(),
                           results.front().shape());
            out.set_unit(a.data().unit());
            for (const Measurement& m : results)
                out.append(m);

            DataArrayCreateInfo info;
            if (rank >= 2)
            {
                // Remaining dimensions survive: the reduced values become
                // dependent data (same demotion as min/max for Independent).
                info.kind = DataArrayKind::kDependent;
                auto it = a.datas().begin();
                for (std::size_t i = 0; i < rank - 1; ++i, ++it)
                    info.datas[it->first] = it->second;
                info.datas[DataArray::kSelf] = std::move(out);

                MultiDimensionSpec spec;
                for (std::size_t i = 0; i < rank - 1; ++i)
                    spec.add_dimension(a.multi_dimension_spec().dim(static_cast<Index>(i)));
                info.multi_dimension_spec = std::move(spec);
            }
            else
            {
                info.kind = DataArrayKind::kIndependent;
                info.datas[DataArray::kSelf] = std::move(out);
                info.multi_dimension_spec = MultiDimensionSpec().add_regular(1);
            }
            return DataArray(std::move(info));
        }, py::arg("fn"),
            "Generic innermost-dimension reduction: fn(numpy group array) -> "
            "scalar is applied to every group of the innermost dimension. "
            "Lowers the rank by one (demotes to a single-value Independent "
            "when no dimension remains). min() == reduce(np.min), "
            "max() == reduce(np.max); any Python fn works (np.mean, custom).")

        // arithmetic -> DataArray (mixed with scalars / arrays / DataArray)
        .def("__add__",  [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return a + b.cast<DataArray>();
            return a + to_measurement(b);
        })
        .def("__radd__", [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return b.cast<DataArray>() + a;
            return to_measurement(b) + a;
        })
        .def("__sub__",  [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return a - b.cast<DataArray>();
            return a - to_measurement(b);
        })
        .def("__rsub__", [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return b.cast<DataArray>() - a;
            return to_measurement(b) - a;
        })
        .def("__mul__",  [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return a * b.cast<DataArray>();
            return a * to_measurement(b);
        })
        .def("__rmul__", [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return b.cast<DataArray>() * a;
            return to_measurement(b) * a;
        })
        .def("__truediv__",  [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return a / b.cast<DataArray>();
            return a / to_measurement(b);
        })
        .def("__rtruediv__", [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return b.cast<DataArray>() / a;
            return to_measurement(b) / a;
        })
        .def("__mod__",  [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return a % b.cast<DataArray>();
            return a % to_measurement(b);
        })
        .def("__rmod__", [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return b.cast<DataArray>() % a;
            return to_measurement(b) % a;
        })
        .def("__pow__",  [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return pow(a, b.cast<DataArray>());
            return pow(a, to_measurement(b));
        })
        .def("__rpow__", [](const DataArray& a, const py::object& b) {
            if (py::isinstance<DataArray>(b)) return pow(b.cast<DataArray>(), a);
            return pow(to_measurement(b), a);
        })
        .def("__neg__", [](const DataArray& a) { return -a; })
        .def("__invert__", [](const DataArray& a) { return ~a; })

        // comparisons -> numpy bool array (numpy semantics)
        .def("__eq__", [](const DataArray& a, const py::object& b) {
            const DataArray r = py::isinstance<DataArray>(b)
                ? (a == b.cast<DataArray>()) : (a == to_measurement(b));
            return series_to_numpy(r.data()).attr("astype")("bool");
        })
        .def("__ne__", [](const DataArray& a, const py::object& b) {
            const DataArray r = py::isinstance<DataArray>(b)
                ? (a != b.cast<DataArray>()) : (a != to_measurement(b));
            return series_to_numpy(r.data()).attr("astype")("bool");
        })
        .def("__lt__", [](const DataArray& a, const py::object& b) {
            const DataArray r = py::isinstance<DataArray>(b)
                ? (a < b.cast<DataArray>()) : (a < to_measurement(b));
            return series_to_numpy(r.data()).attr("astype")("bool");
        })
        .def("__gt__", [](const DataArray& a, const py::object& b) {
            const DataArray r = py::isinstance<DataArray>(b)
                ? (a > b.cast<DataArray>()) : (a > to_measurement(b));
            return series_to_numpy(r.data()).attr("astype")("bool");
        })
        .def("__le__", [](const DataArray& a, const py::object& b) {
            const DataArray r = py::isinstance<DataArray>(b)
                ? (a <= b.cast<DataArray>()) : (a <= to_measurement(b));
            return series_to_numpy(r.data()).attr("astype")("bool");
        })
        .def("__ge__", [](const DataArray& a, const py::object& b) {
            const DataArray r = py::isinstance<DataArray>(b)
                ? (a >= b.cast<DataArray>()) : (a >= to_measurement(b));
            return series_to_numpy(r.data()).attr("astype")("bool");
        })

        .def("__repr__", [](const DataArray& a) {
            return "DataArray(shape=" + py::repr(series_to_numpy(a.data()).attr("shape")).cast<std::string>() +
                   ", dtype=" + py::str(dtype_of(a.data().data_type())).cast<std::string>() +
                   ", unit='" + a.data().unit().to_string() + "')";
        });

    // -----------------------------------------------------------------------
    // Value — unified Measurement-or-DataArray
    // -----------------------------------------------------------------------
    py::class_<Value>(m, "Value",
        "Unified value: either a Measurement or a DataArray. numpy-aligned "
        "metadata and operators; .values exposes the concrete data.")
        .def(py::init<>())
        .def_static("from_numpy", [](const py::object& data, const Unit& unit) {
            py::array arr = py::array::ensure(data);
            if (!arr || arr.ndim() == 0)
            {
                // scalar -> Measurement-backed Value
                return Value(measurement_from_python(data, unit));
            }
            // array -> DataArray-backed Value
            return Value(DataArray::CreateIndependent(series_from_python(data, unit)));
        }, py::arg("data"), py::arg("unit") = Unit(),
            "Build a Value from a scalar / numpy array / nested list.\n"
            "0-d -> Measurement-backed Value; 1-D/2-D/3-D -> DataArray-backed Value.")
        .def_static("Boolean", &Value::Boolean, py::arg("b"))
        .def_static("String", &Value::String, py::arg("s"))
        .def_static("Complex", &Value::Complex, py::arg("v"), py::arg("u") = Unit())
        .def_static("Real", &Value::Real, py::arg("v"), py::arg("u") = Unit())
        .def_static("Integer", &Value::Integer, py::arg("v"), py::arg("u") = Unit())
        .def_static("Vector",
            static_cast<Value(*)(const VecXd&, const Unit&)>(&Value::Vector),
            py::arg("v"), py::arg("u") = Unit())
        .def_static("Vector",
            static_cast<Value(*)(const VecXi&, const Unit&)>(&Value::Vector),
            py::arg("v"), py::arg("u") = Unit())
        .def_static("Vector",
            static_cast<Value(*)(const VecXcd&, const Unit&)>(&Value::Vector),
            py::arg("v"), py::arg("u") = Unit())
        .def_static("Vector", [](const py::list& v) {
            VecXs out(static_cast<Index>(v.size()));
            for (Index i = 0; i < static_cast<Index>(v.size()); ++i)
                out(i) = v[i].cast<std::string>();
            return Value::Vector(out);
        }, py::arg("strings"))
        .def_static("Matrix",
            static_cast<Value(*)(const MatXd&, const Unit&)>(&Value::Matrix),
            py::arg("m"), py::arg("u") = Unit())
        .def_static("Matrix",
            static_cast<Value(*)(const MatXi&, const Unit&)>(&Value::Matrix),
            py::arg("m"), py::arg("u") = Unit())
        .def_static("Matrix",
            static_cast<Value(*)(const MatXcd&, const Unit&)>(&Value::Matrix),
            py::arg("m"), py::arg("u") = Unit())
        .def_static("Matrix", [](const py::list& rows) {
            const Index nr = static_cast<Index>(rows.size());
            const Index nc = nr > 0 ? static_cast<Index>(py::len(rows[0])) : 0;
            MatXs out(nr, nc);
            for (Index r = 0; r < nr; ++r)
                for (Index c = 0; c < nc; ++c)
                    out(r, c) = rows[r].cast<py::list>()[c].cast<std::string>();
            return Value::Matrix(out);
        }, py::arg("strings"))
        .def_static("ArrayReal", &Value::ArrayReal, py::arg("v"), py::arg("u") = Unit())
        .def_static("ArrayInteger", &Value::ArrayInteger, py::arg("v"), py::arg("u") = Unit())
        .def_static("ArrayComplex", &Value::ArrayComplex, py::arg("v"), py::arg("u") = Unit())
        .def_static("ArrayString", &Value::ArrayString, py::arg("v"))
        .def_static("ArrayVector",
            static_cast<Value(*)(const std::vector<VecXd>&, const Unit&)>(&Value::ArrayVector),
            py::arg("rows"), py::arg("u") = Unit())
        .def_static("ArrayVector",
            static_cast<Value(*)(const std::vector<VecXi>&, const Unit&)>(&Value::ArrayVector),
            py::arg("rows"), py::arg("u") = Unit())
        .def_static("ArrayVector",
            static_cast<Value(*)(const std::vector<VecXcd>&, const Unit&)>(&Value::ArrayVector),
            py::arg("rows"), py::arg("u") = Unit())
        .def_static("ArrayMatrix",
            static_cast<Value(*)(const std::vector<MatXd>&, const Unit&)>(&Value::ArrayMatrix),
            py::arg("rows"), py::arg("u") = Unit())
        .def_static("ArrayMatrix",
            static_cast<Value(*)(const std::vector<MatXi>&, const Unit&)>(&Value::ArrayMatrix),
            py::arg("rows"), py::arg("u") = Unit())
        .def_static("ArrayMatrix",
            static_cast<Value(*)(const std::vector<MatXcd>&, const Unit&)>(&Value::ArrayMatrix),
            py::arg("rows"), py::arg("u") = Unit())

        .def_property_readonly("is_measurement", &Value::is_measurement)
        .def_property_readonly("is_data_array", &Value::is_data_array)
        .def_property_readonly("shape", [](const Value& v) -> py::object {
            if (v.is_measurement()) return shape_tuple(v.as_measurement().shape());
            return series_to_numpy(v.as_data_array().data()).attr("shape");
        })
        .def_property_readonly("ndim", [](const Value& v) -> py::object {
            if (v.is_measurement()) return py::int_(static_cast<int>(v.as_measurement().shape().size()));
            return series_to_numpy(v.as_data_array().data()).attr("ndim");
        })
        .def_property_readonly("dtype", [](const Value& v) { return dtype_of(v.data_type()); })
        .def_property_readonly("unit",
            [](const Value& v) -> const Unit& { return v.unit(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly("rows", &Value::rows)
        .def_property_readonly("size", [](const Value& v) {
            return static_cast<py::ssize_t>(v.element_count());
        })
        .def_property_readonly("value", [](const Value& v) -> py::object {
            if (v.is_measurement())
                return measurement_value(v.as_measurement());
            return py::cast(&v.as_data_array(), py::return_value_policy::reference_internal);
        }, "Measurement -> Python scalar / numpy array; DataArray -> DataArray.")
        .def_property_readonly("values", [](const Value& v) -> py::object {
            if (v.is_measurement())
                return measurement_to_numpy(v.as_measurement());
            return series_to_numpy(v.as_data_array().data());
        }, "Always a numpy array (0-d for scalars).")
        .def("as_measurement", [](const Value& v) -> const Measurement& {
            return v.as_measurement();
        }, py::return_value_policy::reference_internal)
        .def("as_data_array", [](const Value& v) -> const DataArray& {
            return v.as_data_array();
        }, py::return_value_policy::reference_internal)

        .def("__array__", [](const Value& v, py::handle dtype, py::handle copy) {
            if (v.is_measurement())
                return apply_dtype(measurement_to_numpy(v.as_measurement()), dtype);
            return apply_dtype(series_to_numpy(v.as_data_array().data()), dtype);
        }, py::arg("dtype") = py::none(), py::arg("copy") = py::none())
        .def("__bool__", [](const Value& v) -> bool {
            if (!v.is_measurement())
                throw py::type_error("the truth value of an array Value is ambiguous");
            const Measurement& m = v.as_measurement();
            if (m.data_kind() != DataKind::kScalar)
                throw py::type_error("the truth value of a non-scalar Value is ambiguous");
            return measurement_truth(m);
        })
        .def("__float__", [](const Value& v) {
            return py::float_(measurement_to_numpy(v.as_measurement()).attr("item")());
        })
        .def("__int__", [](const Value& v) {
            return py::int_(measurement_to_numpy(v.as_measurement()).attr("item")());
        })
        .def("__complex__", [](const Value& v) {
            return py::cast<std::complex<double>>(
                measurement_to_numpy(v.as_measurement()).attr("item")());
        })

        // arithmetic -> Value
        .def("__add__",  [](const Value& a, const py::object& b) { return a + to_value(b); })
        .def("__radd__", [](const Value& a, const py::object& b) { return to_value(b) + a; })
        .def("__sub__",  [](const Value& a, const py::object& b) { return a - to_value(b); })
        .def("__rsub__", [](const Value& a, const py::object& b) { return to_value(b) - a; })
        .def("__mul__",  [](const Value& a, const py::object& b) { return a * to_value(b); })
        .def("__rmul__", [](const Value& a, const py::object& b) { return to_value(b) * a; })
        .def("__truediv__",  [](const Value& a, const py::object& b) { return a / to_value(b); })
        .def("__rtruediv__", [](const Value& a, const py::object& b) { return to_value(b) / a; })
        .def("__mod__",  [](const Value& a, const py::object& b) { return a % to_value(b); })
        .def("__rmod__", [](const Value& a, const py::object& b) { return to_value(b) % a; })
        .def("__pow__",  [](const Value& a, const py::object& b) { return xdataset::pow(a, to_value(b)); })
        .def("__rpow__", [](const Value& a, const py::object& b) { return xdataset::pow(to_value(b), a); })
        .def("__neg__", [](const Value& a) { return -a; })

        // comparisons -> numpy-aligned (bool for scalars, bool arrays otherwise)
        .def("__eq__", [](const Value& a, const py::object& b) -> py::object {
            const Value r = a == to_value(b);
            if (r.is_measurement())
            {
                const Measurement& m = r.as_measurement();
                if (m.data_kind() == DataKind::kScalar && m.data_type() == DataType::kBoolean)
                    return py::bool_(boost::get<bool>(m.storage()));
                return measurement_to_numpy(m).attr("astype")("bool");
            }
            return series_to_numpy(r.as_data_array().data()).attr("astype")("bool");
        })
        .def("__ne__", [](const Value& a, const py::object& b) -> py::object {
            const Value r = a != to_value(b);
            if (r.is_measurement())
            {
                const Measurement& m = r.as_measurement();
                if (m.data_kind() == DataKind::kScalar && m.data_type() == DataType::kBoolean)
                    return py::bool_(boost::get<bool>(m.storage()));
                return measurement_to_numpy(m).attr("astype")("bool");
            }
            return series_to_numpy(r.as_data_array().data()).attr("astype")("bool");
        })
        .def("__lt__", [](const Value& a, const py::object& b) -> py::object {
            const Value r = a < to_value(b);
            if (r.is_measurement())
            {
                const Measurement& m = r.as_measurement();
                if (m.data_kind() == DataKind::kScalar && m.data_type() == DataType::kBoolean)
                    return py::bool_(boost::get<bool>(m.storage()));
                return measurement_to_numpy(m).attr("astype")("bool");
            }
            return series_to_numpy(r.as_data_array().data()).attr("astype")("bool");
        })
        .def("__gt__", [](const Value& a, const py::object& b) -> py::object {
            const Value r = a > to_value(b);
            if (r.is_measurement())
            {
                const Measurement& m = r.as_measurement();
                if (m.data_kind() == DataKind::kScalar && m.data_type() == DataType::kBoolean)
                    return py::bool_(boost::get<bool>(m.storage()));
                return measurement_to_numpy(m).attr("astype")("bool");
            }
            return series_to_numpy(r.as_data_array().data()).attr("astype")("bool");
        })
        .def("__le__", [](const Value& a, const py::object& b) -> py::object {
            const Value r = a <= to_value(b);
            if (r.is_measurement())
            {
                const Measurement& m = r.as_measurement();
                if (m.data_kind() == DataKind::kScalar && m.data_type() == DataType::kBoolean)
                    return py::bool_(boost::get<bool>(m.storage()));
                return measurement_to_numpy(m).attr("astype")("bool");
            }
            return series_to_numpy(r.as_data_array().data()).attr("astype")("bool");
        })
        .def("__ge__", [](const Value& a, const py::object& b) -> py::object {
            const Value r = a >= to_value(b);
            if (r.is_measurement())
            {
                const Measurement& m = r.as_measurement();
                if (m.data_kind() == DataKind::kScalar && m.data_type() == DataType::kBoolean)
                    return py::bool_(boost::get<bool>(m.storage()));
                return measurement_to_numpy(m).attr("astype")("bool");
            }
            return series_to_numpy(r.as_data_array().data()).attr("astype")("bool");
        })

        .def("canonicalized", &Value::canonicalized)
        .def_property_readonly("is_canonicalized", &Value::is_canonicalized)
        .def("format", [](const Value& v, const std::string& name, int max_rows) {
            return v.Format(name, max_rows);
        }, py::arg("name") = "data", py::arg("max_rows") = 32)
        .def("__str__", [](const Value& v) { return v.Format("data", 32); })
        .def("__repr__", [](const Value& v) {
            if (v.is_measurement())
                return "Value(" + v.as_measurement().to_string() + ")";
            return "Value(" + py::repr(py::cast(&v.as_data_array(),
                py::return_value_policy::reference_internal)).cast<std::string>() + ")";
        });

    m.def("pow", [](const Value& base, const Value& exponent) {
        return xdataset::pow(base, exponent);
    }, py::arg("base"), py::arg("exponent"));
}
