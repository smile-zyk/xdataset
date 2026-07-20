// =============================================================================
//  xdataset -- arithmetic operators
// =============================================================================
//
//  All binary arithmetic for the type system lives in this single translation
//  unit, organised in four sections:
//
//    Sec.1  Type promotion helpers  -- promoted_dtype / promoted_kind
//    Sec.2  Measurement  vs  Measurement  -- + - * / pow
//    Sec.3  DataSeries  vs  DataSeries   -- + - * / (delegates to Sec.2)
//    Sec.4  DataSeries  vs  Measurement  -- + - * / pow (delegates to Sec.2)
//    Sec.5  DataArray  vs  DataArray     -- + - * / (delegates to Sec.3)
//    Sec.6  DataArray  vs  Measurement   -- + - * / (delegates to Sec.4)
//    Sec.7  pow(DataArray, Measurement)
//
//  Design principle: the Measurement operators (Sec.2) are the sole "leaf"
//  implementation that touches element-level storage.  DataSeries operators
//  (Sec.3, Sec.4) are thin wrappers: they canonicalise, validate, then delegate
//  per-row work to the Measurement layer.
// =============================================================================

#include "data_series.h"
#include "measurement.h"
#include "data_array.h"

#include <complex>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace xdataset {

// =============================================================================
//  Sec.1  Type promotion helpers
// =============================================================================

// --- dtype promotion: Integer  <  Real  <  Complex -------------------------------

DTypeTag promoted_dtype(DTypeTag a, DTypeTag b) {
    if (a == DTypeTag::kString || b == DTypeTag::kString)
        throw std::invalid_argument(
            "promoted_dtype: string cannot participate in arithmetic");
    if (a == DTypeTag::kComplex || b == DTypeTag::kComplex) return DTypeTag::kComplex;
    if (a == DTypeTag::kReal    || b == DTypeTag::kReal)    return DTypeTag::kReal;
    return DTypeTag::kInteger;
}

// --- DataKind promotion (broadcast) -------------------------------------------
//
//  + - * / pow all share the same kind-compatibility table:
//
//         | Scalar  Vector  Matrix
//    -----+-----------------------
//    Scalar| Scalar  Vector  Matrix   <- broadcast
//    Vector| Vector  Vector  X        <- Vector vs Matrix forbidden
//    Matrix| Matrix  X       Matrix   <- same-shape required for Matrix vs Matrix

DataKind promoted_kind(DataKind a, DataKind b) {
    if (a == DataKind::kMatrix && b == DataKind::kVector) goto incompatible;
    if (a == DataKind::kVector && b == DataKind::kMatrix) goto incompatible;

    if (a == DataKind::kMatrix || b == DataKind::kMatrix) return DataKind::kMatrix;
    if (a == DataKind::kVector || b == DataKind::kVector) return DataKind::kVector;
    return DataKind::kScalar;

incompatible:
    throw std::invalid_argument(
        "promoted_kind: cannot combine Vector with Matrix in arithmetic");
}

// =============================================================================
//  Sec.2  Measurement  vs  Measurement
// =============================================================================
//
//  Each binary operator follows the same pattern:
//    1. validate: reject String, check kind compatibility
//    2. canonicalise both operands (original objects unchanged)
//    3. (+ - only) verify same physical dimension
//    4. derive result kind / dtype / shape / unit
//    5. iterate over elements (with Scalar->Vector/Matrix broadcast)
//    6. assemble result Measurement
//
//  The template meas_binop<trait> encapsulates steps 1-6 so the four concrete
//  operators are just one-liners.

namespace {

// =========================================================================
//  Helpers: element-level access
// =========================================================================

/// Flat element count: Scalar=1, Vector=width, Matrix=rowsxcols.
Index meas_element_count(const Measurement& m) {
    switch (m.kind()) {
        case DataKind::kScalar: return 1;
        case DataKind::kVector: return m.shape()[0];
        case DataKind::kMatrix: return m.shape()[0] * m.shape()[1];
    }
    return 0;  // unreachable
}

/// Read element at flat index, promoting Integer -> double.
double meas_as_double(const Measurement& m, Index i) {
    switch (m.kind()) {
        case DataKind::kScalar:
            return (m.dtype() == DTypeTag::kInteger)
                       ? static_cast<double>(m.as_scalar<int>())
                       : m.as_scalar<double>();
        case DataKind::kVector:
            return (m.dtype() == DTypeTag::kInteger)
                       ? static_cast<double>(m.as_vector<int>()(i))
                       : m.as_vector<double>()(i);
        case DataKind::kMatrix: {
            Index cols = m.shape()[1];
            Index r = i / cols, c = i % cols;
            return (m.dtype() == DTypeTag::kInteger)
                       ? static_cast<double>(m.as_matrix<int>()(r, c))
                       : m.as_matrix<double>()(r, c);
        }
    }
    return 0;
}

/// Read element at flat index, promoting Integer/Real -> complex.
std::complex<double> meas_as_complex(const Measurement& m, Index i) {
    switch (m.kind()) {
        case DataKind::kScalar:
            return (m.dtype() == DTypeTag::kComplex)
                       ? m.as_scalar<std::complex<double> >()
                       : std::complex<double>(meas_as_double(m, i), 0.0);
        case DataKind::kVector:
            return (m.dtype() == DTypeTag::kComplex)
                       ? m.as_vector<std::complex<double> >()(i)
                       : std::complex<double>(meas_as_double(m, i), 0.0);
        case DataKind::kMatrix: {
            Index cols = m.shape()[1];
            Index r = i / cols, c = i % cols;
            return (m.dtype() == DTypeTag::kComplex)
                       ? m.as_matrix<std::complex<double> >()(r, c)
                       : std::complex<double>(meas_as_double(m, i), 0.0);
        }
    }
    return {};
}

/// Read integer element at flat index.
int meas_as_int(const Measurement& m, Index i) {
    switch (m.kind()) {
        case DataKind::kScalar:  return m.as_scalar<int>();
        case DataKind::kVector:  return m.as_vector<int>()(i);
        case DataKind::kMatrix: {
            Index cols = m.shape()[1];
            return m.as_matrix<int>()(i / cols, i % cols);
        }
    }
    return 0;
}

// =========================================================================
//  Helpers: result assembly by kind
// =========================================================================

Measurement make_scalar_real(double v, const Unit& u)            { return {v, u}; }
Measurement make_scalar_int  (int v,    const Unit& u)            { return {v, u}; }
Measurement make_scalar_cplx (std::complex<double> v, const Unit& u) { return {v, u}; }
Measurement make_vector_real (const Eigen::VectorXd& v,  const Unit& u) { return {v, u}; }
Measurement make_vector_int  (const Eigen::VectorXi& v,  const Unit& u) { return {v, u}; }
Measurement make_vector_cplx (const Eigen::VectorXcd& v, const Unit& u) { return {v, u}; }
Measurement make_matrix_real (const Eigen::MatrixXd& v,  const Unit& u) { return {v, u}; }
Measurement make_matrix_int  (const Eigen::MatrixXi& v,  const Unit& u) { return {v, u}; }
Measurement make_matrix_cplx (const Eigen::MatrixXcd& v, const Unit& u) { return {v, u}; }

/// Extract the effective shape from the non-Scalar operand.
inline void get_shape(const Measurement& a, const Measurement& b,
                      Index& rows, Index& cols) {
    if (a.kind() != DataKind::kScalar) {
        rows = a.shape()[0];
        cols = a.shape()[1];
    } else {
        rows = b.shape()[0];
        cols = b.shape()[1];
    }
}

// =========================================================================
//  Operator trait: captures the per-op differences
// =========================================================================

struct MeasOpTraits {
    const char*  op_name;
    bool         require_same_dim;   // + - require it; * / pow do not
    bool         int_div_to_real;    // / needs it; + - * pow do not
    std::function<Unit(const Unit&, const Unit&)> derive_unit;

    // The scalar element operation (all operands are canonicalised double or
    // complex; int-overflow comes from meas_as_int directly).
    double (*real_op)(double, double);
    std::complex<double> (*cplx_op)(std::complex<double>, std::complex<double>);
    int    (*int_op)(int, int);
};

// =========================================================================
//  Generic binary operator for Measurement
// =========================================================================

Measurement meas_binop(const Measurement& lhs, const Measurement& rhs,
                       const MeasOpTraits& tr) {
    // --- Step 1: validation ------------------------------------------------
    if (lhs.dtype() == DTypeTag::kString || rhs.dtype() == DTypeTag::kString)
        throw std::invalid_argument(
            std::string(tr.op_name) + ": string cannot participate in arithmetic");
    promoted_kind(lhs.kind(), rhs.kind());  // throws on VectorxMatrix

    // --- Step 2: canonicalise ----------------------------------------------
    Measurement a = lhs.canonicalized();
    Measurement b = rhs.canonicalized();

    // --- Step 3: unit check (+ / -) ----------------------------------------
    if (tr.require_same_dim && !a.unit().same_dimension(b.unit()))
        throw std::invalid_argument(
            std::string(tr.op_name) + ": unit dimension mismatch [" +
            a.unit().to_string() + "] vs [" + b.unit().to_string() + "]");

    // --- Step 4: result metadata -------------------------------------------
    DataKind  res_kind  = promoted_kind(a.kind(), b.kind());
    DTypeTag  res_dtype = promoted_dtype(a.dtype(), b.dtype());
    if (tr.int_div_to_real &&
        a.dtype() == DTypeTag::kInteger && b.dtype() == DTypeTag::kInteger)
        res_dtype = DTypeTag::kReal;

    Index a_cnt = meas_element_count(a);
    Index b_cnt = meas_element_count(b);
    Index n_elems = std::max(a_cnt, b_cnt);
    bool  a_broad = (a_cnt == 1);   // scalar broadcast?
    bool  b_broad = (b_cnt == 1);

    Unit result_unit = tr.derive_unit(a.unit(), b.unit());

    // --- Step 5: element-wise computation ----------------------------------

    // Complex path
    if (res_dtype == DTypeTag::kComplex) {
        if (res_kind == DataKind::kScalar) {
            return make_scalar_cplx(tr.cplx_op(
                meas_as_complex(a, 0), meas_as_complex(b, 0)), result_unit);
        }
        if (res_kind == DataKind::kVector) {
            Eigen::VectorXcd v(n_elems);
            for (Index i = 0; i < n_elems; ++i)
                v(i) = tr.cplx_op(meas_as_complex(a, a_broad ? 0 : i),
                                   meas_as_complex(b, b_broad ? 0 : i));
            return make_vector_cplx(v, result_unit);
        }
        // Matrix
        Index rows, cols; get_shape(a, b, rows, cols);
        Eigen::MatrixXcd m(rows, cols);
        for (Index r = 0; r < rows; ++r) {
            for (Index c = 0; c < cols; ++c) {
                Index ai = a_broad ? 0 : r * cols + c;
                Index bi = b_broad ? 0 : r * cols + c;
                m(r, c) = tr.cplx_op(meas_as_complex(a, ai),
                                      meas_as_complex(b, bi));
            }
        }
        return make_matrix_cplx(m, result_unit);
    }

    // Integer path (only when both are Integer and op is + - *)
    if (res_dtype == DTypeTag::kInteger) {
        if (res_kind == DataKind::kScalar) {
            return make_scalar_int(tr.int_op(
                meas_as_int(a, 0), meas_as_int(b, 0)), result_unit);
        }
        if (res_kind == DataKind::kVector) {
            Eigen::VectorXi v(n_elems);
            for (Index i = 0; i < n_elems; ++i)
                v(i) = tr.int_op(meas_as_int(a, a_broad ? 0 : i),
                                  meas_as_int(b, b_broad ? 0 : i));
            return make_vector_int(v, result_unit);
        }
        Index rows, cols; get_shape(a, b, rows, cols);
        Eigen::MatrixXi m(rows, cols);
        for (Index r = 0; r < rows; ++r)
            for (Index c = 0; c < cols; ++c) {
                Index ai = a_broad ? 0 : r * cols + c;
                Index bi = b_broad ? 0 : r * cols + c;
                m(r, c) = tr.int_op(meas_as_int(a, ai), meas_as_int(b, bi));
            }
        return make_matrix_int(m, result_unit);
    }

    // Real path (default: all divisions, mixed int/real, pure real)
    if (res_kind == DataKind::kScalar) {
        return make_scalar_real(tr.real_op(
            meas_as_double(a, 0), meas_as_double(b, 0)), result_unit);
    }
    if (res_kind == DataKind::kVector) {
        Eigen::VectorXd v(n_elems);
        for (Index i = 0; i < n_elems; ++i)
            v(i) = tr.real_op(meas_as_double(a, a_broad ? 0 : i),
                               meas_as_double(b, b_broad ? 0 : i));
        return make_vector_real(v, result_unit);
    }
    // Matrix real
    Index rows, cols; get_shape(a, b, rows, cols);
    Eigen::MatrixXd mat(rows, cols);
    for (Index r = 0; r < rows; ++r)
        for (Index c = 0; c < cols; ++c) {
            Index ai = a_broad ? 0 : r * cols + c;
            Index bi = b_broad ? 0 : r * cols + c;
            mat(r, c) = tr.real_op(meas_as_double(a, ai), meas_as_double(b, bi));
        }
    return make_matrix_real(mat, result_unit);
}

// =========================================================================
//  Operator trait factories (compile-time-like: one static instance per op)
// =========================================================================

inline double op_add_d(double a, double b) { return a + b; }
inline int    op_add_i(int a, int b)    { return a + b; }
inline std::complex<double> op_add_c(std::complex<double> a, std::complex<double> b) { return a + b; }

inline double op_sub_d(double a, double b) { return a - b; }
inline int    op_sub_i(int a, int b)    { return a - b; }
inline std::complex<double> op_sub_c(std::complex<double> a, std::complex<double> b) { return a - b; }

inline double op_mul_d(double a, double b) { return a * b; }
inline int    op_mul_i(int a, int b)    { return a * b; }
inline std::complex<double> op_mul_c(std::complex<double> a, std::complex<double> b) { return a * b; }

inline double op_div_d(double a, double b) { return a / b; }
inline int    op_div_i(int, int)       { return 0; }   // never actually used: int/int->Real
inline std::complex<double> op_div_c(std::complex<double> a, std::complex<double> b) { return a / b; }

inline Unit unit_left(const Unit& a, const Unit&) { return a; }

/// Resolve the result unit for addition/subtraction.
/// - Same dimension    -> use a's unit
/// - One dimensionless  -> use the other's unit
/// - Both have different non-dimensionless units -> throw
inline Unit resolve_add_sub_unit(const Unit& a, const Unit& b) {
    if (a.same_dimension(b)) return a;
    if (!a.has_dimension())  return b;
    if (!b.has_dimension())  return a;
    throw std::invalid_argument(
        "operator +/-: unit dimension mismatch [" +
        a.to_string() + "] vs [" + b.to_string() + "]");
}

const MeasOpTraits kMeasAdd = {
    "operator+", false, false, resolve_add_sub_unit, op_add_d, op_add_c, op_add_i};
const MeasOpTraits kMeasSub = {
    "operator-", false, false, resolve_add_sub_unit, op_sub_d, op_sub_c, op_sub_i};
const MeasOpTraits kMeasMul = {
    "operator*", false, false,
    [](const Unit& a, const Unit& b) { return a.multiply_dim(b); },
    op_mul_d, op_mul_c, op_mul_i};
const MeasOpTraits kMeasDiv = {
    "operator/", false, true,
    [](const Unit& a, const Unit& b) { return a.divide_dim(b); },
    op_div_d, op_div_c, op_div_i};

}  // anonymous namespace

// =========================================================================
//  Measurement + - * /
// =========================================================================

Measurement operator+(const Measurement& lhs, const Measurement& rhs) {
    return meas_binop(lhs, rhs, kMeasAdd);
}

Measurement operator-(const Measurement& lhs, const Measurement& rhs) {
    return meas_binop(lhs, rhs, kMeasSub);
}

Measurement operator*(const Measurement& lhs, const Measurement& rhs) {
    return meas_binop(lhs, rhs, kMeasMul);
}

Measurement operator/(const Measurement& lhs, const Measurement& rhs) {
    return meas_binop(lhs, rhs, kMeasDiv);
}

// =========================================================================
//  Measurement pow
// =========================================================================

Measurement pow(const Measurement& base, const Measurement& exponent) {
    // --- validation --------------------------------------------------------
    if (exponent.dtype() == DTypeTag::kString)
        throw std::invalid_argument(
            "pow: exponent cannot be string");
    if (exponent.unit().has_dimension())
        throw std::invalid_argument(
            "pow: exponent must be dimensionless (it carries a physical unit)");
    if (base.dtype() == DTypeTag::kString)
        throw std::invalid_argument(
            "pow: base cannot be string");

    // --- canonicalise -----------------------------------------------------
    Measurement a = base.canonicalized();

    // --- result metadata --------------------------------------------------
    DataKind  res_kind  = promoted_kind(a.kind(), exponent.kind());
    DTypeTag  res_dtype = promoted_dtype(a.dtype(), exponent.dtype());

    Index a_cnt = meas_element_count(a);
    Index e_cnt = meas_element_count(exponent);
    Index n_elems = std::max(a_cnt, e_cnt);
    bool  a_broad = (a_cnt == 1);
    bool  e_broad = (e_cnt == 1);

    Unit result_unit = a.unit();

    // --- complex path -----------------------------------------------------
    if (res_dtype == DTypeTag::kComplex) {
        if (res_kind == DataKind::kScalar)
            return make_scalar_cplx(
                std::pow(meas_as_complex(a, 0), meas_as_complex(exponent, 0)),
                result_unit);
        if (res_kind == DataKind::kVector) {
            Eigen::VectorXcd v(n_elems);
            for (Index i = 0; i < n_elems; ++i)
                v(i) = std::pow(meas_as_complex(a, a_broad ? 0 : i),
                                meas_as_complex(exponent, e_broad ? 0 : i));
            return make_vector_cplx(v, result_unit);
        }
        Index rows, cols; get_shape(a, exponent, rows, cols);
        Eigen::MatrixXcd m(rows, cols);
        for (Index r = 0; r < rows; ++r)
            for (Index c = 0; c < cols; ++c) {
                Index ai = a_broad ? 0 : r * cols + c;
                Index ei = e_broad ? 0 : r * cols + c;
                m(r, c) = std::pow(meas_as_complex(a, ai),
                                   meas_as_complex(exponent, ei));
            }
        return make_matrix_cplx(m, result_unit);
    }

    // --- real path ---------------------------------------------------------
    if (res_kind == DataKind::kScalar)
        return make_scalar_real(
            std::pow(meas_as_double(a, 0), meas_as_double(exponent, 0)),
            result_unit);
    if (res_kind == DataKind::kVector) {
        Eigen::VectorXd v(n_elems);
        for (Index i = 0; i < n_elems; ++i)
            v(i) = std::pow(meas_as_double(a, a_broad ? 0 : i),
                            meas_as_double(exponent, e_broad ? 0 : i));
        return make_vector_real(v, result_unit);
    }
    Index rows, cols; get_shape(a, exponent, rows, cols);
    Eigen::MatrixXd mat(rows, cols);
    for (Index r = 0; r < rows; ++r)
        for (Index c = 0; c < cols; ++c) {
            Index ai = a_broad ? 0 : r * cols + c;
            Index ei = e_broad ? 0 : r * cols + c;
            mat(r, c) = std::pow(meas_as_double(a, ai),
                                 meas_as_double(exponent, ei));
        }
    return make_matrix_real(mat, result_unit);
}

// =============================================================================
//  Sec.3  DataSeries  vs  DataSeries
// =============================================================================
//
//  Thin wrappers.  Each operator:
//    1. Validates row count, rejects String, checks kind compatibility.
//    2. Canonicalises both sides.
//    3. (+ - only) verifies same physical dimension.
//    4. Derives result kind / dtype / shape / unit.
//    5. Iterates rows, delegating each row's computation to
//       the corresponding Measurement operator (Sec.2).

namespace {

// --- row-level validation ----------------------------------------------------
void validate_ds_ds(const DataSeries& a, const DataSeries& b,
                    const char* op_name) {
    if (a.size() != b.size())
        throw std::invalid_argument(
            std::string(op_name) + ": row-count mismatch (" +
            std::to_string(a.size()) + " vs " + std::to_string(b.size()) + ")");
    if (a.dtype() == DTypeTag::kString || b.dtype() == DTypeTag::kString)
        throw std::invalid_argument(
            std::string(op_name) + ": string series cannot participate in arithmetic");
    promoted_kind(a.data_kind(), b.data_kind());  // throws on VectorxMatrix
}

// --- result shape for DataSeries vs DataSeries ----------------------------------
//     Returns the shape of the non-Scalar side; validates same-shape when
//     both sides are non-Scalar (Vector vs Vector, Matrix vs Matrix).
std::vector<Index> ds_ds_result_shape(const DataSeries& a, const DataSeries& b) {
    if (a.data_kind() == DataKind::kScalar && b.data_kind() == DataKind::kScalar)
        return {};
    if (a.data_kind() == DataKind::kScalar) return b.data_shape();
    if (b.data_kind() == DataKind::kScalar) return a.data_shape();

    if (a.data_shape() != b.data_shape())
        throw std::invalid_argument(
            "operator: shape mismatch -- lhs has " +
            std::to_string(a.data_shape()[0]) +
            (a.data_shape().size() > 1 ? "x" + std::to_string(a.data_shape()[1]) : "") +
            ", rhs has " + std::to_string(b.data_shape()[0]) +
            (b.data_shape().size() > 1 ? "x" + std::to_string(b.data_shape()[1]) : ""));
    return a.data_shape();
}

// =========================================================================
//  Generic row-wise application: DataSeries  vs  DataSeries
// =========================================================================
//
//  MeasOp: the Measurement-level operator (e.g. operator+ for Measurements)
//  UnitOp: derives the result unit from the two canonicalised source units

template <typename MeasOp, typename UnitOp>
DataSeries ds_ds_apply(const DataSeries& lhs, const DataSeries& rhs,
                       const char* op_name, MeasOp meas_op, UnitOp unit_op,
                       bool same_dim_required, bool int_div_to_real) {
    validate_ds_ds(lhs, rhs, op_name);

    DataSeries a = lhs.canonicalized();
    DataSeries b = rhs.canonicalized();

    if (same_dim_required && !a.unit().same_dimension(b.unit()))
        throw std::invalid_argument(
            std::string(op_name) + ": unit dimension mismatch [" +
            a.unit().to_string() + "] vs [" + b.unit().to_string() + "]");

    auto res_kind  = promoted_kind(a.data_kind(), b.data_kind());
    auto res_dtype = promoted_dtype(a.dtype(), b.dtype());
    if (int_div_to_real &&
        a.dtype() == DTypeTag::kInteger && b.dtype() == DTypeTag::kInteger)
        res_dtype = DTypeTag::kReal;

    DataSeries result(res_kind, res_dtype, ds_ds_result_shape(a, b));
    result.set_unit(unit_op(a.unit(), b.unit()));

    for (std::size_t i = 0; i < a.size(); ++i)
        result.append(meas_op(
            a.measurement_at(static_cast<Index>(i)),
            b.measurement_at(static_cast<Index>(i))));
    return result;
}

}  // anonymous namespace

DataSeries operator+(const DataSeries& lhs, const DataSeries& rhs) {
    return ds_ds_apply(lhs, rhs, "operator+",
        [](const Measurement& a, const Measurement& b) { return a + b; },
        resolve_add_sub_unit,
        /*same_dim*/ false, /*int_div_real*/ false);
}

DataSeries operator-(const DataSeries& lhs, const DataSeries& rhs) {
    return ds_ds_apply(lhs, rhs, "operator-",
        [](const Measurement& a, const Measurement& b) { return a - b; },
        resolve_add_sub_unit,
        /*same_dim*/ false, /*int_div_real*/ false);
}

DataSeries operator*(const DataSeries& lhs, const DataSeries& rhs) {
    return ds_ds_apply(lhs, rhs, "operator*",
        [](const Measurement& a, const Measurement& b) { return a * b; },
        [](const Unit& a, const Unit& b) { return a.multiply_dim(b); },
        /*same_dim*/ false, /*int_div_real*/ false);
}

DataSeries operator/(const DataSeries& lhs, const DataSeries& rhs) {
    return ds_ds_apply(lhs, rhs, "operator/",
        [](const Measurement& a, const Measurement& b) { return a / b; },
        [](const Unit& a, const Unit& b) { return a.divide_dim(b); },
        /*same_dim*/ false, /*int_div_real*/ true);
}

// =============================================================================
//  Sec.4  DataSeries  vs  Measurement
// =============================================================================
//
//  Identical pattern to Sec.3 except the right operand is a single Measurement
//  broadcast to every row.

namespace {

// --- row-level validation ----------------------------------------------------
void validate_ds_meas(const DataSeries& a, const Measurement& m,
                      const char* op_name) {
    if (a.dtype() == DTypeTag::kString || m.dtype() == DTypeTag::kString)
        throw std::invalid_argument(
            std::string(op_name) + ": string cannot participate in arithmetic");
    promoted_kind(a.data_kind(), m.kind());
}

// --- result shape for DataSeries vs Measurement ---------------------------------
std::vector<Index> ds_meas_result_shape(const DataSeries& ds, const Measurement& m) {
    if (ds.data_kind() == DataKind::kScalar) return m.shape();
    return ds.data_shape();
}

// =========================================================================
//  Generic row-wise application: DataSeries  vs  Measurement
// =========================================================================

template <typename MeasOp, typename UnitOp>
DataSeries ds_meas_apply(const DataSeries& lhs, const Measurement& rhs,
                          const char* op_name, MeasOp meas_op, UnitOp unit_op,
                          bool same_dim_required, bool int_div_to_real) {
    validate_ds_meas(lhs, rhs, op_name);

    DataSeries  a     = lhs.canonicalized();
    Measurement rhs_c = rhs.canonicalized();

    if (same_dim_required && !a.unit().same_dimension(rhs_c.unit()))
        throw std::invalid_argument(
            std::string(op_name) + ": unit dimension mismatch [" +
            a.unit().to_string() + "] vs [" + rhs_c.unit().to_string() + "]");

    auto res_kind  = promoted_kind(a.data_kind(), rhs_c.kind());
    auto res_dtype = promoted_dtype(a.dtype(), rhs_c.dtype());
    if (int_div_to_real &&
        a.dtype() == DTypeTag::kInteger && rhs_c.dtype() == DTypeTag::kInteger)
        res_dtype = DTypeTag::kReal;

    DataSeries result(res_kind, res_dtype, ds_meas_result_shape(a, rhs_c));
    result.set_unit(unit_op(a.unit(), rhs_c.unit()));

    for (std::size_t i = 0; i < a.size(); ++i)
        result.append(meas_op(
            a.measurement_at(static_cast<Index>(i)), rhs_c));
    return result;
}

}  // anonymous namespace

DataSeries operator+(const DataSeries& lhs, const Measurement& rhs) {
    return ds_meas_apply(lhs, rhs, "operator+",
        [](const Measurement& a, const Measurement& b) { return a + b; },
        resolve_add_sub_unit,
        /*same_dim*/ false, /*int_div_real*/ false);
}

DataSeries operator-(const DataSeries& lhs, const Measurement& rhs) {
    return ds_meas_apply(lhs, rhs, "operator-",
        [](const Measurement& a, const Measurement& b) { return a - b; },
        resolve_add_sub_unit,
        /*same_dim*/ false, /*int_div_real*/ false);
}

DataSeries operator*(const DataSeries& lhs, const Measurement& rhs) {
    return ds_meas_apply(lhs, rhs, "operator*",
        [](const Measurement& a, const Measurement& b) { return a * b; },
        [](const Unit& a, const Unit& b) { return a.multiply_dim(b); },
        /*same_dim*/ false, /*int_div_real*/ false);
}

DataSeries operator/(const DataSeries& lhs, const Measurement& rhs) {
    return ds_meas_apply(lhs, rhs, "operator/",
        [](const Measurement& a, const Measurement& b) { return a / b; },
        [](const Unit& a, const Unit& b) { return a.divide_dim(b); },
        /*same_dim*/ false, /*int_div_real*/ true);
}

// =============================================================================
//  Sec.5  pow(DataSeries, Measurement)
// =============================================================================
//
//  Exponent must be dimensionless, non-String, and (for now) Scalar.
//  For each row we delegate to pow(Measurement, Measurement) from Sec.2.

DataSeries pow(const DataSeries& base, const Measurement& exp) {
    // --- validation ---------------------------------------------------------
    if (exp.dtype() == DTypeTag::kString)
        throw std::invalid_argument("pow: exponent cannot be string");
    if (exp.unit().has_dimension())
        throw std::invalid_argument(
            "pow: exponent must be dimensionless (it carries a physical unit)");
    if (exp.kind() != DataKind::kScalar)
        throw std::invalid_argument(
            "pow: exponent in DataSeries::pow must be a scalar Measurement");
    if (base.dtype() == DTypeTag::kString)
        throw std::invalid_argument(
            "pow: string series cannot participate in arithmetic");

    // --- canonicalise -------------------------------------------------------
    DataSeries  a     = base.canonicalized();
    Measurement exp_c = exp.canonicalized();

    auto res_dtype = promoted_dtype(base.dtype(), exp.dtype());
    DataSeries result(base.data_kind(), res_dtype, base.data_shape());

    // Dimensionless exponent preserves the base unit.
    result.set_unit(a.unit());

    // --- per-row computation ------------------------------------------------
    for (std::size_t i = 0; i < a.size(); ++i)
        result.append(xdataset::pow(
            a.measurement_at(static_cast<Index>(i)), exp_c));
    return result;
}

// =============================================================================
//  Sec.5  DataArray  vs  DataArray
// =============================================================================
//
//  Thin wrappers that:
//    1. Validate MultiDimensionSpec compatibility (same rank, dims, sizes).
//    2. Validate indep_datas have the same count of independents.
//    3. Delegate to the corresponding DataSeries operator (Sec.3).
//    4. Assemble the result DataArray: name = placeholder, data = result DS,
//       indep_datas / multi_dimension_spec / kind = inherited from lhs.
//
//  The template array_array_op encapsulates the boilerplate shared by all
//  four operators (the only differences are the DataSeries-level op, the
//  same-dimension-required flag, and the int-div->real flag).

namespace {

// ---- validation ----------------------------------------------------------

/// Verify two MultiDimensionSpecs are structurally identical.
void validate_spec_compatible(const MultiDimensionSpec& a, const MultiDimensionSpec& b,
                               const char* op_name) {
    if (a.rank() != b.rank())
        throw std::invalid_argument(
            std::string(op_name) + ": MultiDimensionSpec rank mismatch (" +
            std::to_string(a.rank()) + " vs " + std::to_string(b.rank()) + ")");

    const auto& da = a.dims();
    const auto& db = b.dims();
    for (std::size_t i = 0; i < da.size(); ++i) {
        if (da[i].is_regular() != db[i].is_regular()) {
            throw std::invalid_argument(
                std::string(op_name) + ": dimension " + std::to_string(i) +
                " regular/ragged mismatch");
        }
        if (da[i].is_regular()) {
            if (da[i].regular_size() != db[i].regular_size())
                throw std::invalid_argument(
                    std::string(op_name) + ": dimension " + std::to_string(i) +
                    " size mismatch (" + std::to_string(da[i].regular_size()) +
                    " vs " + std::to_string(db[i].regular_size()) + ")");
        } else {
            if (da[i].ragged_sizes() != db[i].ragged_sizes())
                throw std::invalid_argument(
                    std::string(op_name) + ": dimension " + std::to_string(i) +
                    " ragged sizes differ");
        }
    }
}

/// Verify two DataArrays have compatible independent-variable sets.
/// For independent arrays the last indep name (the array's own name) is
/// excluded since it naturally differs between two different arrays.
void validate_indeps_compatible(const DataArray& a, const DataArray& b,
                                 const char* op_name) {
    auto na = a.indep_names();
    auto nb = b.indep_names();
    if (na.size() != nb.size())
        throw std::invalid_argument(
            std::string(op_name) + ": independent variable count mismatch (" +
            std::to_string(na.size()) + " vs " + std::to_string(nb.size()) + ")");

    std::size_t n = na.size();
    if (a.kind() == DataArrayKind::kIndependent && b.kind() == DataArrayKind::kIndependent)
        --n;  // skip the last name (the array's own name)

    for (std::size_t i = 0; i < n; ++i) {
        if (na[i] != nb[i])
            throw std::invalid_argument(
                std::string(op_name) + ": independent variable names differ at position " +
                std::to_string(i) + " (" + na[i] + " vs " + nb[i] + ")");
    }
}

/// Build the result name for a binary DataArray op.
inline std::string result_name(const DataArray&, const DataArray&,
                               const char*) {
    return DataArray::kUnnamed;
}

// =========================================================================
//  Generic DataArray  vs  DataArray operator
// =========================================================================

template <typename DataSeriesOp>
DataArray array_array_op(const DataArray& lhs, const DataArray& rhs,
                          const char* op_name, const char* op_symbol,
                          DataSeriesOp ds_op) {
    // --- validation --------------------------------------------------------
    validate_spec_compatible(lhs.multi_dimension_spec(), rhs.multi_dimension_spec(),
                              op_name);
    validate_indeps_compatible(lhs, rhs, op_name);

    // --- delegate to DataSeries --------------------------------------------
    DataSeries result_ds = ds_op(lhs.data(), rhs.data());

    // --- assemble result DataArray -----------------------------------------
    DataArrayCreateInfo info;
    info.name                = result_name(lhs, rhs, op_symbol);
    info.data                = std::move(result_ds);
    info.indep_datas         = lhs.indep_datas();         // inherit from lhs
    info.multi_dimension_spec = lhs.multi_dimension_spec(); // inherit from lhs
    info.kind                = lhs.kind();                // inherit from lhs
    return DataArray(std::move(info));
}

}  // anonymous namespace

DataArray operator+(const DataArray& lhs, const DataArray& rhs) {
    return array_array_op(lhs, rhs, "operator+", "+",
        [](const DataSeries& a, const DataSeries& b) { return a + b; });
}

DataArray operator-(const DataArray& lhs, const DataArray& rhs) {
    return array_array_op(lhs, rhs, "operator-", "-",
        [](const DataSeries& a, const DataSeries& b) { return a - b; });
}

DataArray operator*(const DataArray& lhs, const DataArray& rhs) {
    return array_array_op(lhs, rhs, "operator*", "*",
        [](const DataSeries& a, const DataSeries& b) { return a * b; });
}

DataArray operator/(const DataArray& lhs, const DataArray& rhs) {
    return array_array_op(lhs, rhs, "operator/", "/",
        [](const DataSeries& a, const DataSeries& b) { return a / b; });
}

// =============================================================================
//  Sec.6  DataArray  vs  Measurement
// =============================================================================
//
//  Identical pattern to Sec.5 except the right operand is a single Measurement
//  broadcast to every row of the DataArray's DataSeries.

namespace {

template <typename DataSeriesMeasOp>
DataArray array_meas_op(const DataArray& lhs, const Measurement& rhs,
                         const char* op_name, const char* op_symbol,
                         DataSeriesMeasOp ds_meas_op) {
    // --- delegate to DataSeries --------------------------------------------
    DataSeries result_ds = ds_meas_op(lhs.data(), rhs);

    // --- assemble result DataArray -----------------------------------------
    DataArrayCreateInfo info;
    info.name                 = DataArray::kUnnamed;
    info.data                 = std::move(result_ds);
    info.indep_datas          = lhs.indep_datas();
    info.multi_dimension_spec = lhs.multi_dimension_spec();
    info.kind                 = lhs.kind();
    return DataArray(std::move(info));
}

}  // anonymous namespace

DataArray operator+(const DataArray& lhs, const Measurement& rhs) {
    return array_meas_op(lhs, rhs, "operator+", "+",
        [](const DataSeries& a, const Measurement& b) { return a + b; });
}

DataArray operator-(const DataArray& lhs, const Measurement& rhs) {
    return array_meas_op(lhs, rhs, "operator-", "-",
        [](const DataSeries& a, const Measurement& b) { return a - b; });
}

DataArray operator*(const DataArray& lhs, const Measurement& rhs) {
    return array_meas_op(lhs, rhs, "operator*", "*",
        [](const DataSeries& a, const Measurement& b) { return a * b; });
}

DataArray operator/(const DataArray& lhs, const Measurement& rhs) {
    return array_meas_op(lhs, rhs, "operator/", "/",
        [](const DataSeries& a, const Measurement& b) { return a / b; });
}

// =============================================================================
//  Sec.7  pow(DataArray, Measurement)
// =============================================================================

DataArray pow(const DataArray& base, const Measurement& exp) {
    DataSeries result_ds = xdataset::pow(base.data(), exp);

    DataArrayCreateInfo info;
    info.name                 = DataArray::kUnnamed;
    info.data                 = std::move(result_ds);
    info.indep_datas          = base.indep_datas();
    info.multi_dimension_spec = base.multi_dimension_spec();
    info.kind                 = base.kind();
    return DataArray(std::move(info));
}

}  // namespace xdataset
