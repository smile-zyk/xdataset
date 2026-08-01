// =============================================================================
//  xdataset -- operation framework
// =============================================================================
//
//  Pipeline:
//    Operate() derives result metadata (shape, rows, dtype, unit) from
//    operands, packs them into ExecContextInfo, then dispatches to the
//    execute callback registered in OpTraits.
//
//  The execute callback is responsible for performing the actual
//  computation.  For binary arithmetic, ExecuteBinaryArith flattens
//  all operands to typed T* buffers, computes broadcast plans, runs
//  a unified loop, and converts the flat output back to Value.
//
//  Broadcast plans (RowBroadcastPlan, ShapeBroadcastPlan) are computed
//  inside the execute callbacks that need them; they do not appear in
//  the Operate pipeline.

#include "operation.h"
#include "data_series.h"
#include "data_array.h"

#include <Eigen/LU>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace xdataset {

// =========================================================================
//  Internal types (not exposed via operation.h)
// =========================================================================

enum class OpCategory {
    kPow,
    kNot, kBitNot, kNegate,
    kMul, kDiv, kMod,
    kAdd, kSub,
    kShl, kShr,
    kLt, kGt, kLe, kGe,
    kEq, kNeq,
    kBitAnd, kBitXor, kBitOr,
    kAnd, kOr,
    kConditional,
    kMatrix, kSweep
};

struct RowBroadcastPlan {
    Index              result_size;
    std::vector<bool>  broadcast;

    static RowBroadcastPlan Compute(const std::vector<Index>& sizes);
};

struct OperandBroadcastShapeInfo {
    Index elements;
    Index cols;
    bool  broadcast_row;
    bool  broadcast_col;
};

struct ShapeBroadcastPlan {
    DataShape          result_shape;
    Index              result_elements;
    Index              result_cols;
    std::vector<OperandBroadcastShapeInfo> ops;

    static ShapeBroadcastPlan Make(const std::vector<DataShape>& operand_shapes,
                                    const DataShape& result);
    Index MapFlatIndex(Index result_flat, int k) const;
};

struct ExecContextInfo {
    OpCategory         op;
    Index              rows;
    DataShape          shape;
    DataType           dtype;
    Unit               unit;
};

template <typename T>
using ElemOp = T (*)(T, T);

template <typename T>
using UnaryOp = T (*)(T);

typedef DataShape (*DeriveShapeFunc)(const std::vector<DataShape>& operand_shapes);
typedef DataType (*DeriveDtypeFunc)(const std::vector<DataType>& dtypes);
typedef Unit     (*DeriveUnitFunc)(const std::vector<Unit>& units);
typedef Index    (*DeriveRowsFunc)(const std::vector<Index>& rows);
typedef Value    (*ExecuteFunc)(const ExecContextInfo& info,
                                const std::vector<Value>& ops);

enum Arity : Index {
    kVariadic = -1
};

struct OpTraits {
    OpCategory      op;
    Index           arity;
    DeriveShapeFunc derive_shape;
    DeriveRowsFunc  derive_rows;
    DeriveDtypeFunc derive_dtype;
    DeriveUnitFunc  derive_unit;
    ExecuteFunc     execute;
};

// =========================================================================
//  RowBroadcastPlan::Compute
// =========================================================================

RowBroadcastPlan RowBroadcastPlan::Compute(const std::vector<Index>& sizes) {
    RowBroadcastPlan plan;
    Index r = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        Index s = sizes[i];
        if (s == 1) continue;
        if (r == 1) { r = s; continue; }
        if (s != r)
            throw std::invalid_argument(
                "broadcast size mismatch (" + std::to_string(r) +
                " vs " + std::to_string(s) + ") at index " +
                std::to_string(i));
    }
    plan.result_size = r;
    for (size_t i = 0; i < sizes.size(); ++i)
        plan.broadcast.push_back(sizes[i] == 1 && r > 1);
    return plan;
}

// =========================================================================
//  ShapeBroadcastPlan::Make
// =========================================================================

ShapeBroadcastPlan ShapeBroadcastPlan::Make(const std::vector<DataShape>& operand_shapes,
                     const DataShape& result) {
    ShapeBroadcastPlan sp;
    DataKind rk = result.kind();
    sp.result_shape    = result;
    sp.result_elements = (rk == DataKind::kScalar) ? 1
                       : (rk == DataKind::kVector) ? result[0]
                       : result[0] * result[1];

    if (rk == DataKind::kMatrix)
        sp.result_cols = result[1];
    else if (rk == DataKind::kVector)
        sp.result_cols = result[0];
    else
        sp.result_cols = 1;

    for (size_t i = 0; i < operand_shapes.size(); ++i) {
        DataKind k = operand_shapes[i].kind();
        const auto& s = operand_shapes[i];

        OperandBroadcastShapeInfo op;
        op.elements = (k == DataKind::kScalar) ? 1
                    : (k == DataKind::kVector) ? s[0]
                    : s[0] * s[1];

        bool bc_row = false, bc_col = false;
        Index cols = 1;

        if (k == DataKind::kScalar) {
            bc_row = true;
            bc_col = true;
            cols   = 1;

        } else if (k == DataKind::kVector) {
            Index w = s[0];
            if (rk == DataKind::kVector) {
                bc_col = (w == 1 && sp.result_cols > 1);
                cols   = w;
            } else /* result is Matrix */ {
                // row-vector [1, w] -> broadcast rows, columns must match
                bc_row = true;
                bc_col = (w == 1 && result[1] > 1);
                cols   = w;
            }

        } else /* kMatrix */ {
            cols   = s[1];
            bc_row = (s[0] == 1 && result[0] > 1);
            bc_col = (s[1] == 1 && result[1] > 1);
        }

        op.cols = cols;
        op.broadcast_row = bc_row;
        op.broadcast_col = bc_col;
        sp.ops.push_back(op);
    }

    return sp;
}

// =========================================================================
//  ShapeBroadcastPlan::MapFlatIndex
// =========================================================================

Index ShapeBroadcastPlan::MapFlatIndex(Index result_flat, int k) const {
    const OperandBroadcastShapeInfo& op = ops[static_cast<size_t>(k)];
    if (op.elements == 1) return 0;

    // Vectors are treated as single-row matrices (1 row, w cols)
    Index row = 0, col = result_flat;
    if (result_cols > 1 && result_elements != result_cols) {
        row = result_flat / result_cols;
        col = result_flat % result_cols;
    }

    Index r = op.broadcast_row ? 0 : row;
    Index c = op.broadcast_col ? 0 : col;
    return r * op.cols + c;
}

// =========================================================================
//  Operate
// =========================================================================

Value Operate(const std::vector<Value>& operands, const OpTraits& traits) {
    // --- arity check ---------------------------------------------------
    if (traits.arity != Arity::kVariadic) {
        Index n = static_cast<Index>(operands.size());
        if (n != traits.arity) {
            throw std::invalid_argument(
                std::string("arity mismatch: expected ") +
                std::to_string(traits.arity) + " operand(s), got " +
                std::to_string(n));
        }
    }

    // Canonicalize operands (absorb scale multipliers, convert to base SI)
    std::vector<Value> canonical_ops;
    canonical_ops.reserve(operands.size());
    for (size_t i = 0; i < operands.size(); ++i) {
        canonical_ops.push_back(operands[i].canonicalized());
    }

    // Extract per-operand metadata from canonicalized operands
    std::vector<DataShape> operand_shapes;
    std::vector<Index>     row_counts;
    std::vector<DataType>  dtypes;
    std::vector<Unit>      units;

    for (size_t i = 0; i < canonical_ops.size(); ++i) {
        operand_shapes.push_back(canonical_ops[i].data_shape());
        row_counts.push_back(canonical_ops[i].rows());
        dtypes.push_back(canonical_ops[i].data_type());
        units.push_back(canonical_ops[i].unit());
    }

    // Derive result metadata from canonicalized operands
    DataShape shape       = traits.derive_shape(operand_shapes);
    Index     rows        = traits.derive_rows(row_counts);
    DataType  dtype       = traits.derive_dtype(dtypes);
    Unit      unit        = traits.derive_unit(units);

    // Pack context
    ExecContextInfo info;
    info.op    = traits.op;
    info.rows  = rows;
    info.shape = shape;
    info.dtype = dtype;
    info.unit  = unit;

    return traits.execute(info, canonical_ops);
}

// =========================================================================
//  Derive callbacks
// =========================================================================

// -- DeriveShapeBroadcast --

DataShape DeriveShapeBroadcast(const std::vector<DataShape>& operand_shapes) {
    // --- result kind ---
    DataKind res_kind = DataKind::kScalar;
    for (size_t i = 0; i < operand_shapes.size(); ++i) {
        DataKind k = operand_shapes[i].kind();
        if (k == DataKind::kMatrix)
            res_kind = DataKind::kMatrix;
        else if (k == DataKind::kVector && res_kind != DataKind::kMatrix)
            res_kind = DataKind::kVector;
    }

    if (res_kind == DataKind::kScalar)
        return DataShape{};

    if (res_kind == DataKind::kVector) {
        Index w = 1;
        for (size_t i = 0; i < operand_shapes.size(); ++i) {
            if (operand_shapes[i].kind() == DataKind::kScalar) continue;
            Index sw = operand_shapes[i][0];
            if (sw == 1) continue;
            if (w == 1) { w = sw; continue; }
            if (sw != w)
                throw std::invalid_argument(
                    "vector width mismatch (" + std::to_string(w) +
                    " vs " + std::to_string(sw) + ")");
        }
        return DataShape{w};
    }

    // Matrix: row & col independently
    Index r = 1, c = 1;
    for (size_t i = 0; i < operand_shapes.size(); ++i) {
        Index op_r = 1, op_c = 1;
        DataKind k = operand_shapes[i].kind();
        const auto& s = operand_shapes[i];

        if      (k == DataKind::kScalar) { op_r = 1; op_c = 1; }
        else if (k == DataKind::kVector) { op_r = 1; op_c = s[0]; }  // 1 row, w cols
        else /* kMatrix */               { op_r = s[0]; op_c = s[1]; }

        if (op_r == 1) { /* broadcast */ }
        else if (r == 1) { r = op_r; }
        else if (op_r != r)
            throw std::invalid_argument(
                "row dim mismatch (" + std::to_string(r) +
                " vs " + std::to_string(op_r) + ")");

        if (op_c == 1) { /* broadcast */ }
        else if (c == 1) { c = op_c; }
        else if (op_c != c)
            throw std::invalid_argument(
                "col dim mismatch (" + std::to_string(c) +
                " vs " + std::to_string(op_c) + ")");
    }
    return DataShape{r, c};
}

// -- DeriveShapeMatrix --

DataShape DeriveShapeMatrix(const std::vector<DataShape>& operand_shapes) {
    if (operand_shapes.empty())
        throw std::invalid_argument("concat: empty input");

    const DataKind k0 = operand_shapes[0].kind();
    const DataShape& s0 = operand_shapes[0];
    for (size_t i = 1; i < operand_shapes.size(); ++i) {
        if (operand_shapes[i].kind() != k0)
            throw std::invalid_argument(
                "concat: kind mismatch at index " + std::to_string(i));
        if (operand_shapes[i] != s0)
            throw std::invalid_argument(
                "concat: shape mismatch at index " + std::to_string(i));
    }

    const Index N = static_cast<Index>(operand_shapes.size());

    DataShape result;
    if (k0 == DataKind::kScalar)
        result = DataShape{N};
    else if (k0 == DataKind::kVector)
        result = DataShape{N, s0[0]};
    else
        throw std::invalid_argument("concat: cannot concat matrices");

    return result;
}

// =========================================================================
//  Helper: effective (r, c) for matrix multiplication
// =========================================================================
//  Scalar -> (1,1)  Vector[w] -> (1,w)  Matrix[r,c] -> (r,c)

namespace {

std::pair<Index, Index> EffectiveRC(const DataShape& s) {
    DataKind k = s.kind();
    if (k == DataKind::kScalar) return {1, 1};
    if (k == DataKind::kVector) return {1, s[0]};
    return {s[0], s[1]};
}

DataShape MakeShapeRC(Index r, Index c) {
    if (r == 1 && c == 1) return DataShape{};
    if (r == 1)           return DataShape{c};
    if (c == 1)           return DataShape{r};
    return DataShape{r, c};
}

}  // anonymous namespace

// -- DeriveShapeMul --

DataShape DeriveShapeMul(const std::vector<DataShape>& operand_shapes) {
    // If any operand is scalar, fall back to element-wise broadcast
    if (operand_shapes[0].kind() == DataKind::kScalar ||
        operand_shapes[1].kind() == DataKind::kScalar)
        return DeriveShapeBroadcast(operand_shapes);

    // Both non-scalar: matrix multiplication (Vector always 1xw)
    auto rcA = EffectiveRC(operand_shapes[0]);
    auto rcB = EffectiveRC(operand_shapes[1]);
    Index rA = rcA.first, cA = rcA.second;
    Index rB = rcB.first, cB = rcB.second;

    if (cA != rB)
        throw std::invalid_argument(
            "matmul inner dim mismatch: (" + std::to_string(rA) + "x" +
            std::to_string(cA) + ") x (" + std::to_string(rB) + "x" +
            std::to_string(cB) + ")");

    return MakeShapeRC(rA, cB);
}

// -- DeriveShapeDiv --

DataShape DeriveShapeDiv(const std::vector<DataShape>& operand_shapes) {
    // If RHS is scalar, fall back to element-wise broadcast
    if (operand_shapes[1].kind() == DataKind::kScalar)
        return DeriveShapeBroadcast(operand_shapes);

    // RHS non-scalar: A / B = A x inv(B)
    // pinv(B) has effective shape (cB, rB)
    auto rcA = EffectiveRC(operand_shapes[0]);
    auto rcB = EffectiveRC(operand_shapes[1]);
    Index rA = rcA.first, cA = rcA.second;
    Index rB = rcB.first, cB = rcB.second;

    // RHS must be square for true inverse; non-square uses pseudo-inverse.
    // For square RHS, invertibility will be verified at runtime.
    if (rB != cB)
        throw std::invalid_argument(
            "RHS matrix must be square for division (got " +
            std::to_string(rB) + "x" + std::to_string(cB) + ")");

    // A x inv(B): (rA, cA) x (cB, rB), inner dim cA == cB
    if (cA != cB)
        throw std::invalid_argument(
            "A(cols) must equal B(cols) for division: (" + std::to_string(rA) + "x" +
            std::to_string(cA) + ") / (" + std::to_string(rB) + "x" +
            std::to_string(cB) + ")");

    return MakeShapeRC(rA, rB);
}

// -- DeriveRowsBroadcast --

Index DeriveRowsBroadcast(const std::vector<Index>& rows) {
    return RowBroadcastPlan::Compute(rows).result_size;
}

Index DeriveRowsSum(const std::vector<Index>& rows) {
    Index total = 0;
    for (size_t i = 0; i < rows.size(); ++i) total += rows[i];
    return total;
}

// -- DeriveDtypePromote --

DataType DeriveDtypePromote(const std::vector<DataType>& dtypes) {
    DataType res = DataType::kInteger;
    for (size_t i = 0; i < dtypes.size(); ++i) {
        DataType dt = dtypes[i];
        if (dt == DataType::kString)
            throw std::invalid_argument(
                "arithmetic: string operand not allowed");
        // Boolean is Measurement-only; normalize to Integer for computation
        if (dt == DataType::kBoolean)
            dt = DataType::kInteger;
        if (dt == DataType::kComplex)
            res = DataType::kComplex;
        else if (dt == DataType::kReal && res != DataType::kComplex)
            res = DataType::kReal;
    }
    return res;
}

// -- DeriveDtypeDiv --

DataType DeriveDtypeDiv(const std::vector<DataType>& dtypes) {
    DataType res = DeriveDtypePromote(dtypes);
    if (res == DataType::kInteger) res = DataType::kReal;
    return res;
}

// -- DeriveDtypeMod --

DataType DeriveDtypeMod(const std::vector<DataType>& dtypes) {
    // Only int and double; Bool normalizes to int.  Complex/string throw.
    bool has_real = false;
    for (size_t i = 0; i < dtypes.size(); ++i) {
        DataType dt = dtypes[i];
        if (dt == DataType::kBoolean) dt = DataType::kInteger;
        if (dt == DataType::kReal)    { has_real = true; continue; }
        if (dt == DataType::kInteger) continue;
        throw std::invalid_argument(
            "mod: unsupported type " + std::to_string(static_cast<int>(dt)));
    }
    return has_real ? DataType::kReal : DataType::kInteger;
}

// -- DeriveDtypePow --

DataType DeriveDtypePow(const std::vector<DataType>& dtypes) {
    DataType res = DeriveDtypePromote(dtypes);
    if (res == DataType::kInteger) res = DataType::kReal;
    return res;
}

// -- DeriveDtypeBitwise --

DataType DeriveDtypeBitwise(const std::vector<DataType>& dtypes) {
    for (size_t i = 0; i < dtypes.size(); ++i) {
        DataType dt = dtypes[i];
        if (dt == DataType::kBoolean) continue;  // bool OK, normalizes to int
        if (dt == DataType::kInteger) continue;
        throw std::invalid_argument(
            "bitwise: int operand required, got type " +
            std::to_string(static_cast<int>(dt)));
    }
    return DataType::kInteger;
}

// -- DeriveDtypeCmp -- infer comparison type from operand types, result always int

DataType DeriveDtypeCmp(const std::vector<DataType>& dtypes) {
    DataType res = DataType::kInteger;
    for (size_t i = 0; i < dtypes.size(); ++i) {
        DataType dt = dtypes[i];
        if (dt == DataType::kBoolean) dt = DataType::kInteger;
        if (dt == DataType::kString) { res = DataType::kString; break; }
        if (dt == DataType::kComplex && res != DataType::kString)
            res = DataType::kComplex;
        else if (dt == DataType::kReal && res != DataType::kComplex && res != DataType::kString)
            res = DataType::kReal;
    }
    return res;
}

DataType DeriveDtypeForceInt(const std::vector<DataType>& /*dtypes*/) {
    return DataType::kInteger;
}

// -- DeriveDtypeMerge --

DataType DeriveDtypeMerge(const std::vector<DataType>& dtypes) {
    if (dtypes.empty()) return DataType::kInteger;
    bool all_string = true, any_string = false;
    DataType res = DataType::kInteger;
    for (size_t i = 0; i < dtypes.size(); ++i) {
        DataType dt = dtypes[i];
        if (dt == DataType::kBoolean) dt = DataType::kInteger;
        if (dt == DataType::kString) {
            any_string = true;
            continue;
        }
        all_string = false;
        if (dt == DataType::kComplex)
            res = DataType::kComplex;
        else if (dt == DataType::kReal && res != DataType::kComplex)
            res = DataType::kReal;
    }
    if (any_string && !all_string)
        throw std::invalid_argument("concat/sweep: cannot mix string with numeric types");
    return all_string ? DataType::kString : res;
}

// -- DeriveUnitSameDim --

Unit DeriveUnitSameDim(const std::vector<Unit>& units) {
    if (units.empty()) return Unit();
    Unit res = units[0];
    for (size_t i = 1; i < units.size(); ++i) {
        if (res.same_dimension(units[i])) continue;
        if (!res.has_dimension()) { res = units[i]; continue; }
        if (!units[i].has_dimension()) continue;
        throw std::invalid_argument("unit dimension mismatch");
    }
    return res;
}

// -- DeriveUnitMul --

Unit DeriveUnitMul(const std::vector<Unit>& units) {
    Unit res = units[0];
    for (size_t i = 1; i < units.size(); ++i)
        res = res * units[i];
    return res;
}

// -- DeriveUnitDiv --

Unit DeriveUnitDiv(const std::vector<Unit>& units) {
    Unit res = units[0];
    for (size_t i = 1; i < units.size(); ++i)
        res = res / units[i];
    return res;
}

// -- DeriveUnitDimless --

Unit DeriveUnitDimless(const std::vector<Unit>& /*units*/) {
    return Unit();
}

// -- DeriveUnitFirst --

Unit DeriveUnitFirst(const std::vector<Unit>& units) {
    return units[0];
}

// =========================================================================
//  Element-wise operators
// =========================================================================
//
//  Arithmetic, comparison, and logical element functions.  Each is
//  instantiated for double, int, and std::complex<double>.

namespace {

template <typename T> inline T op_add(T a, T b) { return a + b; }
template <typename T> inline T op_sub(T a, T b) { return a - b; }
template <typename T> inline T op_mul(T a, T b) { return a * b; }
template <typename T> inline T op_div(T a, T b) { return a / b; }
template <typename T> inline T op_mod(T a, T b) {
    (void)a; (void)b;
    throw std::invalid_argument("mod not supported for this type");
}
template <> inline int op_mod<int>(int a, int b) {
    return a % b;
}
template <> inline double op_mod<double>(double a, double b) {
    return std::fmod(a, b);
}

template <typename T> inline T op_pow(T a, T b) {
    (void)a; (void)b;
    throw std::invalid_argument("pow not supported for this type");
}
template <> inline double op_pow<double>(double a, double b) {
    return std::pow(a, b);
}
template <> inline std::complex<double> op_pow<std::complex<double>>(
    std::complex<double> a, std::complex<double> b) {
    return std::pow(a, b);
}

// Numeric cmp: compare at actual type, return 0/1
// complex uses abs() for < <= > >=
template <typename T> inline int op_cmp_eq(T a, T b) { return a == b ? 1 : 0; }
template <typename T> inline int op_cmp_ne(T a, T b) { return a != b ? 1 : 0; }
template <typename T> inline int op_cmp_lt(T a, T b) { return a <  b ? 1 : 0; }
template <typename T> inline int op_cmp_gt(T a, T b) { return a >  b ? 1 : 0; }
template <typename T> inline int op_cmp_le(T a, T b) { return a <= b ? 1 : 0; }
template <typename T> inline int op_cmp_ge(T a, T b) { return a >= b ? 1 : 0; }

template <> inline int op_cmp_lt<std::complex<double>>(std::complex<double> a, std::complex<double> b) { return std::abs(a) <  std::abs(b) ? 1 : 0; }
template <> inline int op_cmp_gt<std::complex<double>>(std::complex<double> a, std::complex<double> b) { return std::abs(a) >  std::abs(b) ? 1 : 0; }
template <> inline int op_cmp_le<std::complex<double>>(std::complex<double> a, std::complex<double> b) { return std::abs(a) <= std::abs(b) ? 1 : 0; }
template <> inline int op_cmp_ge<std::complex<double>>(std::complex<double> a, std::complex<double> b) { return std::abs(a) >= std::abs(b) ? 1 : 0; }

// String cmp — non-template to avoid copy overhead
inline int str_cmp_eq(const std::string& a, const std::string& b) { return a == b ? 1 : 0; }
inline int str_cmp_ne(const std::string& a, const std::string& b) { return a != b ? 1 : 0; }
inline int str_cmp_lt(const std::string& a, const std::string& b) { return a <  b ? 1 : 0; }
inline int str_cmp_gt(const std::string& a, const std::string& b) { return a >  b ? 1 : 0; }
inline int str_cmp_le(const std::string& a, const std::string& b) { return a <= b ? 1 : 0; }
inline int str_cmp_ge(const std::string& a, const std::string& b) { return a >= b ? 1 : 0; }

// Get cmp element-op by T and category
template <typename T> int (*GetCmpOp(OpCategory cat))(T, T);
template <> inline int (*GetCmpOp<int>(OpCategory cat))(int, int) {
    switch (cat) {
        case OpCategory::kEq:  return op_cmp_eq<int>;  case OpCategory::kNeq: return op_cmp_ne<int>;
        case OpCategory::kLt:  return op_cmp_lt<int>;  case OpCategory::kGt:  return op_cmp_gt<int>;
        case OpCategory::kLe:  return op_cmp_le<int>;  case OpCategory::kGe:  return op_cmp_ge<int>;
        default: throw std::invalid_argument("not a comparison op");
    }
}
template <> inline int (*GetCmpOp<double>(OpCategory cat))(double, double) {
    switch (cat) {
        case OpCategory::kEq:  return op_cmp_eq<double>; case OpCategory::kNeq: return op_cmp_ne<double>;
        case OpCategory::kLt:  return op_cmp_lt<double>; case OpCategory::kGt:  return op_cmp_gt<double>;
        case OpCategory::kLe:  return op_cmp_le<double>; case OpCategory::kGe:  return op_cmp_ge<double>;
        default: throw std::invalid_argument("not a comparison op");
    }
}
template <> inline int (*GetCmpOp<std::complex<double>>(OpCategory cat))(std::complex<double>, std::complex<double>) {
    switch (cat) {
        case OpCategory::kEq:  return op_cmp_eq<std::complex<double>>; case OpCategory::kNeq: return op_cmp_ne<std::complex<double>>;
        case OpCategory::kLt:  return op_cmp_lt<std::complex<double>>; case OpCategory::kGt:  return op_cmp_gt<std::complex<double>>;
        case OpCategory::kLe:  return op_cmp_le<std::complex<double>>; case OpCategory::kGe:  return op_cmp_ge<std::complex<double>>;
        default: throw std::invalid_argument("not a comparison op");
    }
}

inline int (*GetStrCmpOp(OpCategory cat))(const std::string&, const std::string&) {
    switch (cat) {
        case OpCategory::kEq:  return str_cmp_eq; case OpCategory::kNeq: return str_cmp_ne;
        case OpCategory::kLt:  return str_cmp_lt; case OpCategory::kGt:  return str_cmp_gt;
        case OpCategory::kLe:  return str_cmp_le; case OpCategory::kGe:  return str_cmp_ge;
        default: throw std::invalid_argument("not a comparison op");
    }
}

template <typename T> inline T op_and(T a, T b) {
    return static_cast<T>(
        (static_cast<int>(a) && static_cast<int>(b)) ? 1 : 0);
}
template <typename T> inline T op_or(T a, T b) {
    return static_cast<T>(
        (static_cast<int>(a) || static_cast<int>(b)) ? 1 : 0);
}

template <typename T> inline T op_bitand(T /*a*/, T /*b*/) {
    throw std::invalid_argument("bitwise & not supported for this type");
}
template <> inline int op_bitand<int>(int a, int b) { return a & b; }

template <typename T> inline T op_bitor(T /*a*/, T /*b*/) {
    throw std::invalid_argument("bitwise | not supported for this type");
}
template <> inline int op_bitor<int>(int a, int b) { return a | b; }

template <typename T> inline T op_bitxor(T /*a*/, T /*b*/) {
    throw std::invalid_argument("bitwise ^ not supported for this type");
}
template <> inline int op_bitxor<int>(int a, int b) { return a ^ b; }

template <typename T> inline T op_shl(T /*a*/, T /*b*/) {
    throw std::invalid_argument("shift << not supported for this type");
}
template <> inline int op_shl<int>(int a, int b) {
    return (b >= 0) ? (a << b) : (a >> (-b));
}

template <typename T> inline T op_shr(T /*a*/, T /*b*/) {
    throw std::invalid_argument("shift >> not supported for this type");
}
template <> inline int op_shr<int>(int a, int b) {
    return (b >= 0) ? (a >> b) : (a << (-b));
}

template <typename T>
ElemOp<T> GetArithElemOp(OpCategory cat) {
    switch (cat) {
        case OpCategory::kAdd: return op_add<T>;
        case OpCategory::kSub: return op_sub<T>;
        case OpCategory::kMul: return op_mul<T>;
        case OpCategory::kDiv: return op_div<T>;
        case OpCategory::kMod: return op_mod<T>;
        case OpCategory::kPow: return op_pow<T>;
        default: throw std::invalid_argument("not an arithmetic op");
    }
}

template <typename T>
ElemOp<T> GetLogicalElemOp(OpCategory cat) {
    switch (cat) {
        case OpCategory::kAnd: return op_and<T>;
        case OpCategory::kOr:  return op_or<T>;
        default: throw std::invalid_argument("not a logical op");
    }
}

template <typename T>
ElemOp<T> GetBitwiseElemOp(OpCategory cat) {
    switch (cat) {
        case OpCategory::kBitAnd: return op_bitand<T>;
        case OpCategory::kBitOr:  return op_bitor<T>;
        case OpCategory::kBitXor: return op_bitxor<T>;
        default: throw std::invalid_argument("not a bitwise op");
    }
}

template <typename T>
ElemOp<T> GetShiftElemOp(OpCategory cat) {
    switch (cat) {
        case OpCategory::kShl: return op_shl<T>;
        case OpCategory::kShr: return op_shr<T>;
        default: throw std::invalid_argument("not a shift op");
    }
}

}  // anonymous namespace

// =========================================================================
//  Unary element ops
// =========================================================================

namespace {

template <typename T> inline T op_negate(T a) { return -a; }

template <typename T> inline T op_not(T /*a*/) {
    throw std::invalid_argument("logical not not supported for this type");
}
template <> inline int op_not<int>(int a) {
    return (a == 0) ? 1 : 0;
}

template <typename T> inline T op_bitnot(T /*a*/) {
    throw std::invalid_argument("bitwise not not supported for this type");
}
template <> inline int op_bitnot<int>(int a) { return ~a; }

}  // anonymous namespace

// =========================================================================
//  ExecBinaryLoop -- core flat-buffer loop
// =========================================================================
//
//  Row-level and cell-level broadcast are driven by the two plans.
//  This function does not depend on ExecContextInfo.

template <typename T, typename Out = T>
void ExecBinaryLoop(Index rows,
                     const RowBroadcastPlan&   row_plan,
                     const ShapeBroadcastPlan& shape_plan,
                     const T* l_ptr, Index l_stride,
                     const T* r_ptr, Index r_stride,
                     Out* out,
                     Out (*elem_op)(T, T))
{
    Index out_stride = shape_plan.result_elements;

    for (Index i = 0; i < rows; ++i) {
        Index l_row_off = (row_plan.broadcast[0] ? 0 : i) * l_stride;
        Index r_row_off = (row_plan.broadcast[1] ? 0 : i) * r_stride;
        Index o_off     = i * out_stride;

        for (Index j = 0; j < shape_plan.result_elements; ++j) {
            Index lj = shape_plan.MapFlatIndex(j, 0);
            Index rj = shape_plan.MapFlatIndex(j, 1);
            out[o_off + j] = elem_op(
                l_ptr[l_row_off + lj],
                r_ptr[r_row_off + rj]);
        }
    }
}

// =========================================================================
//  Output helpers
// =========================================================================

namespace {

/// Reconstruct a Measurement from a flat typed buffer and a DataShape.
template <typename T>
Measurement MakeMeasFromFlat(const T* data,
                              const DataShape& shape,
                              const Unit& unit) {
    DataKind dk = shape.kind();

    if (dk == DataKind::kScalar)
        return Measurement(data[0], unit);

    if (dk == DataKind::kVector) {
        Index w = shape[0];
        Eigen::Matrix<T, 1, Eigen::Dynamic> v(w);
        for (Index i = 0; i < w; ++i)
            v(i) = data[i];
        return Measurement(v, unit);
    }

    // Matrix
    Index r = shape[0];
    Index c = shape[1];
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(r, c);
    for (Index i = 0; i < r; ++i)
        for (Index j = 0; j < c; ++j)
            m(i, j) = data[i * c + j];
    return Measurement(m, unit);
}

Value MakeArrayFromFlat(std::unique_ptr<DataSeries> ds, const DataArray& src) {
    DataArrayCreateInfo arr_info;
    arr_info.datas                    = src.datas();
    arr_info.datas[DataArray::kSelf]  = std::move(*ds);
    arr_info.multi_dimension_spec     = src.multi_dimension_spec();
    arr_info.kind                     = src.data_kind();
    return Value(DataArray(std::move(arr_info)));
}

/// Binary ops: when the output is a DataArray, choose which operand's
/// metadata (MultiDimensionSpec, DataArrayKind) to inherit.
static const DataArray* SelectOutputSource(bool l_meas, bool r_meas,
                                            const std::vector<Value>& ops) {
    if (!l_meas && !r_meas) return &ops[0].as_data_array();
    if (l_meas && !r_meas) return &ops[1].as_data_array();
    if (!l_meas && r_meas) return &ops[0].as_data_array();
    return nullptr;
}

}  // anonymous namespace

// =========================================================================
//  FlatInput<T> -- typed flat pointer + lifetime for one binary operand
// =========================================================================
//
//  Measurement: always creates a single-row DataSeries, promoted to T.
//  DataArray  : borrows the underlying DataSeries (no copy) when dtype
//               already matches T; copies + promotes otherwise.

namespace {

template <typename T>
struct FlatInput {
    std::unique_ptr<DataSeries> owner;
    const T*                    ptr;
    Index                       stride;   // T-elements per logical row

    /// Construct a single-row DataSeries from a Boolean Measurement in the target dtype.
    static std::unique_ptr<DataSeries> MakeBoolSeries(const Measurement& m) {
        auto ds = std::unique_ptr<DataSeries>(
            new DataSeries(m.data_kind(), DataTypeOf<T>::tag, m.shape()));
        ds->resize(1);

        T val = static_cast<T>(m.as_scalar<bool>() ? 1 : 0);

        if (m.data_kind() == DataKind::kScalar) {
            ds->scalar_at<T>(0) = val;
        } else if (m.data_kind() == DataKind::kVector) {
            for (Index j = 0; j < m.shape()[0]; ++j)
                ds->vector_at<T>(0)(j) = val;
        } else {
            for (Index r = 0; r < m.shape()[0]; ++r)
                for (Index c = 0; c < m.shape()[1]; ++c)
                    ds->matrix_at<T>(0)(r, c) = val;
        }
        return ds;
    }

    static FlatInput Acquire(const Value& v);
};

template <typename T>
FlatInput<T> FlatInput<T>::Acquire(const Value& v) {
    if (v.is_measurement()) {
        FlatInput fi;
        const Measurement& m = v.as_measurement();

        // Boolean is Measurement-only; DataSeries doesn't support it.
        // Build a single-row DataSeries directly in the target dtype.
        DataType src_dtype = m.data_type();
        if (src_dtype == DataType::kBoolean) {
            fi.owner = MakeBoolSeries(m);
            fi.ptr    = fi.owner->template contiguous_data<T>();
            fi.stride = static_cast<Index>(fi.owner->element_count());
            return fi;
        }

        fi.owner = std::unique_ptr<DataSeries>(
            new DataSeries(m.data_kind(), m.data_type(), m.shape()));
        fi.owner->append(m);

        DataType target = DataTypeOf<T>::tag;
        if (m.data_type() != target) {
            fi.owner = std::unique_ptr<DataSeries>(
                new DataSeries(fi.owner->promoted_data_type(target)));
        }
        fi.ptr    = fi.owner->template contiguous_data<T>();
        fi.stride = static_cast<Index>(fi.owner->element_count());
        return fi;
    }

    // DataArray
    FlatInput fi;
    const DataSeries& src = v.as_data_array().data();
    if (src.data_type() == DataTypeOf<T>::tag) {
        // borrow directly -- no copy
        fi.ptr    = src.contiguous_data<T>();
        fi.stride = static_cast<Index>(src.element_count());
    } else {
        DataType target = DataTypeOf<T>::tag;
        fi.owner = std::unique_ptr<DataSeries>(
            new DataSeries(src.promoted_data_type(target)));
        fi.ptr    = fi.owner->template contiguous_data<T>();
        fi.stride = static_cast<Index>(fi.owner->element_count());
    }
    return fi;
}

}  // anonymous namespace

// =========================================================================
//  ExecBinaryArithT -- binary arithmetic entry point
// =========================================================================
//
//  Extract operand metadata, compute broadcast plans, flatten inputs,
//  allocate output, run the unified loop, and convert back to Value.

template <typename T>
Value ExecBinaryArithT(const ExecContextInfo& info,
                        const std::vector<Value>& ops,
                        ElemOp<T> elem_op)
{
    bool l_meas = ops[0].is_measurement();
    bool r_meas = ops[1].is_measurement();

    // --- 提取 per-operand shape/rows ---
    DataShape l_shape = ops[0].data_shape();
    DataShape r_shape = ops[1].data_shape();
    std::vector<DataShape> op_shapes = {l_shape, r_shape};

    Index l_rows = ops[0].rows();
    Index r_rows = ops[1].rows();
    std::vector<Index> row_counts = {l_rows, r_rows};

    // --- 计算广播 plan ---
    ShapeBroadcastPlan shape_plan = ShapeBroadcastPlan::Make(op_shapes, info.shape);
    RowBroadcastPlan   row_plan   = RowBroadcastPlan::Compute(row_counts);

    // ====================================
    //  Step 1: acquire typed flat pointers
    // ====================================
    auto l_in    = FlatInput<T>::Acquire(ops[0]);
    auto r_in    = FlatInput<T>::Acquire(ops[1]);
    const T* l_ptr    = l_in.ptr;
    const T* r_ptr    = r_in.ptr;
    Index    l_stride = l_in.stride;
    Index    r_stride = r_in.stride;

    // --- zero-divisor check (div/mod, both Meas scalar only) ---
    if (l_meas && r_meas && info.shape.kind() == DataKind::kScalar &&
        (info.op == OpCategory::kDiv || info.op == OpCategory::kMod)) {
        if (r_ptr[0] == T(0))
            throw std::invalid_argument("division by zero");
    }

    // ====================================
    //  Step 2: 分配输出
    // ====================================
    const DataArray* out_src = SelectOutputSource(l_meas, r_meas, ops);

    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataTypeOf<T>::tag, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(info.rows));
    T* out = out_ds->mutable_contiguous_data<T>();

    // ====================================
    //  Step 3: 统一核心循环
    // ====================================
    ExecBinaryLoop(info.rows, row_plan, shape_plan,
                   l_ptr, l_stride, r_ptr, r_stride, out, elem_op);

    // ====================================
    //  Step 4: 输出 -> Value
    // ====================================
    if (l_meas && r_meas) {
        return Value(MakeMeasFromFlat(out, info.shape, info.unit));
    } else {
        return MakeArrayFromFlat(std::move(out_ds), *out_src);
    }
}

// =========================================================================
//  Public execute callbacks
// =========================================================================

Value ExecuteBinaryArith(const ExecContextInfo& info,
                          const std::vector<Value>& ops) {
    switch (info.dtype) {
        case DataType::kComplex:
            return ExecBinaryArithT<std::complex<double>>(
                info, ops, GetArithElemOp<std::complex<double>>(info.op));
        case DataType::kReal:
            return ExecBinaryArithT<double>(
                info, ops, GetArithElemOp<double>(info.op));
        case DataType::kInteger:
            return ExecBinaryArithT<int>(
                info, ops, GetArithElemOp<int>(info.op));
        default:
            throw std::invalid_argument(
                "unsupported dtype for arithmetic");
    }
}

// =========================================================================
//  ExecBinaryCmpT -- compare at type T, output int 0/1
// =========================================================================

template <typename T>
Value ExecBinaryCmpT(const ExecContextInfo& info,
                      const std::vector<Value>& ops)
{
    bool l_meas = ops[0].is_measurement();
    bool r_meas = ops[1].is_measurement();

    DataShape l_shape = ops[0].data_shape();
    DataShape r_shape = ops[1].data_shape();
    std::vector<DataShape> op_shapes = {l_shape, r_shape};

    Index l_rows = ops[0].rows();
    Index r_rows = ops[1].rows();
    std::vector<Index> row_counts = {l_rows, r_rows};

    ShapeBroadcastPlan shape_plan = ShapeBroadcastPlan::Make(op_shapes, info.shape);
    RowBroadcastPlan   row_plan   = RowBroadcastPlan::Compute(row_counts);

    auto l_in    = FlatInput<T>::Acquire(ops[0]);
    auto r_in    = FlatInput<T>::Acquire(ops[1]);
    const T* l_ptr    = l_in.ptr;
    const T* r_ptr    = r_in.ptr;
    Index    l_stride = l_in.stride;
    Index    r_stride = r_in.stride;

    const DataArray* out_src = SelectOutputSource(l_meas, r_meas, ops);

    // Output is int (0/1), using same shape/kind as result
    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataType::kInteger, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(info.rows));
    int* out = out_ds->mutable_contiguous_data<int>();

    auto elem_op = GetCmpOp<T>(info.op);
    Index out_stride = shape_plan.result_elements;

    ExecBinaryLoop<T, int>(info.rows, row_plan, shape_plan,
                           l_ptr, l_stride, r_ptr, r_stride, out, elem_op);

    if (l_meas && r_meas) {
        // Scalar Meas -> upgrade to Boolean
        if (info.shape.kind() == DataKind::kScalar)
            return Value(Measurement::Boolean(out[0] != 0));

        return Value(MakeMeasFromFlat(out, info.shape, info.unit));
    } else {
        return MakeArrayFromFlat(std::move(out_ds), *out_src);
    }
}

// -- String path for Cmp ---

static Value ExecBinaryCmpString(const ExecContextInfo& info,
                                  const std::vector<Value>& ops) {
    // Read strings directly; comparison at string type, output int
    bool l_meas = ops[0].is_measurement();
    bool r_meas = ops[1].is_measurement();

    DataShape l_shape = ops[0].data_shape();
    DataShape r_shape = ops[1].data_shape();
    std::vector<DataShape> op_shapes = {l_shape, r_shape};

    Index l_rows = ops[0].rows();
    Index r_rows = ops[1].rows();
    std::vector<Index> row_counts = {l_rows, r_rows};

    ShapeBroadcastPlan shape_plan = ShapeBroadcastPlan::Make(op_shapes, info.shape);
    RowBroadcastPlan   row_plan   = RowBroadcastPlan::Compute(row_counts);

    // Build flat string arrays (no FlatInput — strings handled separately)
    Index l_stride = static_cast<Index>(l_shape.element_count());
    Index r_stride = static_cast<Index>(r_shape.element_count());
    Index result_rows = info.rows;
    Index out_stride  = shape_plan.result_elements;

    // Pre-read all strings into flat vectors
    std::vector<std::string> l_flat, r_flat;
    auto read_flat = [](const Value& v, Index rows, Index stride,
                        std::vector<std::string>& out) {
        out.resize(static_cast<std::size_t>(rows * stride));
        if (v.is_measurement()) {
            const Measurement& m = v.as_measurement();
            DataKind dk = m.data_kind();
            if (dk == DataKind::kScalar) {
                std::string s = m.as_scalar<std::string>();
                for (Index i = 0; i < rows * stride; ++i)
                    out[static_cast<std::size_t>(i)] = s;
            } else if (dk == DataKind::kVector) {
                auto vec = m.as_vector<std::string>();
                for (Index i = 0; i < rows; ++i)
                    for (Index j = 0; j < stride; ++j)
                        out[static_cast<std::size_t>(i * stride + j)] = vec(j);
            } else {
                auto mat = m.as_matrix<std::string>();
                Index cols = m.shape()[1];
                for (Index i = 0; i < rows; ++i)
                    for (Index j = 0; j < stride; ++j)
                        out[static_cast<std::size_t>(i * stride + j)] = mat(j / cols, j % cols);
            }
        } else {
            const DataSeries& ds = v.as_data_array().data();
            DataKind dk = ds.data_kind();
            for (Index i = 0; i < rows; ++i) {
                Index src_row = i;
                if (dk == DataKind::kScalar)
                    out[static_cast<std::size_t>(i * stride)] = ds.scalar_at<std::string>(src_row);
                else if (dk == DataKind::kVector)
                    for (Index j = 0; j < stride; ++j)
                        out[static_cast<std::size_t>(i * stride + j)] = ds.vector_at<std::string>(src_row)(j);
                else {
                    Index cols = ds.data_shape()[1];
                    for (Index j = 0; j < stride; ++j)
                        out[static_cast<std::size_t>(i * stride + j)] = ds.matrix_at<std::string>(src_row)(j / cols, j % cols);
                }
            }
        }
    };
    read_flat(ops[0], l_rows, l_stride, l_flat);
    read_flat(ops[1], r_rows, r_stride, r_flat);

    // Compare strings directly
    auto elem_op = GetStrCmpOp(info.op);

    const DataArray* out_src = SelectOutputSource(l_meas, r_meas, ops);

    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataType::kInteger, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(info.rows));
    int* out = out_ds->mutable_contiguous_data<int>();

    for (Index i = 0; i < info.rows; ++i) {
        Index l_row_off = (row_plan.broadcast[0] ? 0 : i) * l_stride;
        Index r_row_off = (row_plan.broadcast[1] ? 0 : i) * r_stride;
        Index o_off     = i * out_stride;

        for (Index j = 0; j < shape_plan.result_elements; ++j) {
            Index lj = shape_plan.MapFlatIndex(j, 0);
            Index rj = shape_plan.MapFlatIndex(j, 1);
            out[o_off + j] = elem_op(
                l_flat[static_cast<std::size_t>(l_row_off + lj)],
                r_flat[static_cast<std::size_t>(r_row_off + rj)]);
        }
    }

    if (l_meas && r_meas) {
        if (info.shape.kind() == DataKind::kScalar)
            return Value(Measurement::Boolean(out[0] != 0));
        return Value(MakeMeasFromFlat(out, info.shape, info.unit));
    } else {
        return MakeArrayFromFlat(std::move(out_ds), *out_src);
    }
}

// =========================================================================
//  Public execute callbacks -- Cmp and Logic
// =========================================================================

Value ExecuteBinaryCmp(const ExecContextInfo& info,
                        const std::vector<Value>& ops) {
    // info.dtype = comparison type (from DeriveDtypeCmp).
    // Result is always int 0/1.
    // For scalar Meas×Meas the int is upgraded to Boolean.

    switch (info.dtype) {
        case DataType::kString:
            if (ops[0].data_type() != DataType::kString || ops[1].data_type() != DataType::kString)
                throw std::invalid_argument("comparison: cannot mix string with numeric");
            return ExecBinaryCmpString(info, ops);
        case DataType::kComplex:
            return ExecBinaryCmpT<std::complex<double>>(info, ops);
        case DataType::kReal:
            return ExecBinaryCmpT<double>(info, ops);
        default:
            return ExecBinaryCmpT<int>(info, ops);
    }
}

Value ExecuteBinaryLogical(const ExecContextInfo& info,
                            const std::vector<Value>& ops) {
    // Logical ops (AND/OR) first convert both operands to int via as_logical()
    // (non-zero → 1), then apply the logical element op in int domain.
    // Both Measurement and Scalar shape → upgrade to Boolean.

    // Build int operands via as_logical()
    auto make_logical = [](const Value& v) -> Value {
        if (v.is_measurement()) {
            const Measurement& m = v.as_measurement();
            if (m.data_type() == DataType::kBoolean) {
                return Value(Measurement(static_cast<int>(m.as_scalar<bool>() ? 1 : 0)));
            }
            DataSeries ds(m.data_kind(), m.data_type(), m.shape());
            ds.append(m);
            auto logical_ds = std::unique_ptr<DataSeries>(new DataSeries(ds.as_logical()));
            return Value(MakeMeasFromFlat(logical_ds->contiguous_data<int>(),
                        m.shape(), m.unit()));
        } else {
            auto logical_ds = std::unique_ptr<DataSeries>(new DataSeries(v.as_data_array().data().as_logical()));
            return MakeArrayFromFlat(std::move(logical_ds), v.as_data_array());
        }
    };
    Value li = make_logical(ops[0]);
    Value ri = make_logical(ops[1]);

    Value result = ExecBinaryArithT<int>(info, {li, ri},
        GetLogicalElemOp<int>(info.op));

    if (ops[0].is_measurement() && ops[1].is_measurement() &&
        info.shape.kind() == DataKind::kScalar) {
        return Value(Measurement::Boolean(result.as_measurement().as_scalar<int>() != 0));
    }
    return result;
}

// =========================================================================
//  ExecuteBinaryBitwise -- int-only element-wise bitwise ops
// =========================================================================

Value ExecuteBinaryBitwise(const ExecContextInfo& info,
                            const std::vector<Value>& ops) {
    return ExecBinaryArithT<int>(info, ops,
        GetBitwiseElemOp<int>(info.op));
}

// =========================================================================
//  ExecuteBinaryShift -- int-only element-wise shift ops
// =========================================================================

Value ExecuteBinaryShift(const ExecContextInfo& info,
                          const std::vector<Value>& ops) {
    return ExecBinaryArithT<int>(info, ops,
        GetShiftElemOp<int>(info.op));
}

// =========================================================================
//  ExecuteMatrix ({} generator) - stack operands with row broadcast
// =========================================================================
//
//  Output: all Measurement → Measurement, otherwise DataArray.

template <typename T>
static Value ExecMatrixT(const ExecContextInfo& info,
                          const std::vector<Value>& ops) {
    const Index N = static_cast<Index>(ops.size());

    bool all_meas = true;
    for (size_t i = 0; i < ops.size(); ++i)
        if (!ops[i].is_measurement()) { all_meas = false; break; }

    std::vector<Index> row_counts;
    for (size_t i = 0; i < ops.size(); ++i)
        row_counts.push_back(ops[i].rows());
    RowBroadcastPlan row_plan = RowBroadcastPlan::Compute(row_counts);

    std::vector<FlatInput<T>> inputs;
    for (size_t i = 0; i < ops.size(); ++i)
        inputs.push_back(FlatInput<T>::Acquire(ops[i]));

    Index cell_elems = static_cast<Index>(inputs[0].stride);
    Index result_rows = info.rows;

    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataTypeOf<T>::tag, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(result_rows));
    T* out = out_ds->mutable_contiguous_data<T>();

    Index row_stride = cell_elems * N;
    for (Index r = 0; r < result_rows; ++r) {
        Index out_off = r * row_stride;
        for (Index k = 0; k < N; ++k) {
            Index op_row = (ops[static_cast<size_t>(k)].is_measurement())
                           ? 0
                           : (row_plan.broadcast[static_cast<size_t>(k)] ? 0 : r);
            const T* src = inputs[static_cast<size_t>(k)].ptr + op_row * inputs[static_cast<size_t>(k)].stride;
            for (Index j = 0; j < cell_elems; ++j)
                out[out_off + k * cell_elems + j] = src[j];
        }
    }

    if (all_meas) {
        return Value(MakeMeasFromFlat(out, info.shape, info.unit));
    }

    // Preserve metadata from first DataArray operand
    const DataArray* tmpl = nullptr;
    for (size_t i = 0; i < ops.size(); ++i)
        if (ops[i].is_data_array()) { tmpl = &ops[i].as_data_array(); break; }
    if (tmpl)
        return MakeArrayFromFlat(std::move(out_ds), *tmpl);

    return Value(DataArray::CreateIndependent(std::move(*out_ds)));
}

// -- String path: no contiguous_data, access Measurement/DataSeries directly ---

static Value ExecMatrixString(const ExecContextInfo& info,
                               const std::vector<Value>& ops) {
    const Index N = static_cast<Index>(ops.size());

    bool all_meas = true;
    for (size_t i = 0; i < ops.size(); ++i)
        if (!ops[i].is_measurement()) { all_meas = false; break; }

    std::vector<Index> row_counts;
    for (size_t i = 0; i < ops.size(); ++i)
        row_counts.push_back(ops[i].rows());
    RowBroadcastPlan row_plan = RowBroadcastPlan::Compute(row_counts);

    Index cell_elems = ops[0].data_shape().element_count();
    Index result_rows = info.rows;
    Index total = result_rows * cell_elems * N;

    // Build flat string vector, then construct final Measurement/DataSeries
    std::vector<std::string> flat(static_cast<std::size_t>(total));

    Index row_stride = cell_elems * N;
    for (Index r = 0; r < result_rows; ++r) {
        Index out_off = r * row_stride;
        for (Index k = 0; k < N; ++k) {
            Index op_row = row_plan.broadcast[static_cast<size_t>(k)] ? 0 : r;
            Index base = static_cast<Index>(static_cast<std::size_t>(out_off) + static_cast<std::size_t>(k) * static_cast<std::size_t>(cell_elems));

            if (ops[static_cast<size_t>(k)].is_measurement()) {
                const Measurement& m = ops[static_cast<size_t>(k)].as_measurement();
                DataKind dk = m.data_kind();
                if (dk == DataKind::kScalar) {
                    std::string s = m.as_scalar<std::string>();
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = s;
                } else if (dk == DataKind::kVector) {
                    auto vec = m.as_vector<std::string>();
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = vec(j);
                } else {
                    auto mat = m.as_matrix<std::string>();
                    Index cols = m.shape()[1];
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = mat(j / cols, j % cols);
                }
            } else {
                const DataSeries& ds = ops[static_cast<size_t>(k)].as_data_array().data();
                DataKind dk = ds.data_kind();
                if (dk == DataKind::kScalar) {
                    std::string s = ds.scalar_at<std::string>(op_row);
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = s;
                } else if (dk == DataKind::kVector) {
                    auto vec = ds.vector_at<std::string>(op_row);
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = vec(j);
                } else {
                    auto mat = ds.matrix_at<std::string>(op_row);
                    Index cols = ds.data_shape()[1];
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = mat(j / cols, j % cols);
                }
            }
        }
    }

    if (all_meas) {
        // Single Measurement output
        DataKind dk = info.shape.kind();
        if (dk == DataKind::kVector) {
            Index w = info.shape[0];
            VecXs vec(w);
            for (Index i = 0; i < w; ++i)
                vec(i) = std::move(flat[static_cast<std::size_t>(i)]);
            return Value(Measurement::Vector(vec));
        }
        Index rows = info.shape[0], cols = info.shape[1];
        MatXs mat(rows, cols);
        for (Index i = 0; i < rows; ++i)
            for (Index j = 0; j < cols; ++j)
                mat(i, j) = std::move(flat[static_cast<std::size_t>(i * cols + j)]);
        return Value(Measurement::Matrix(mat));
    }

    // DataArray output
    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataType::kString, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(result_rows));
    for (Index r = 0; r < result_rows; ++r) {
        Index base = r * row_stride;
        if (info.shape.kind() == DataKind::kScalar)
            out_ds->scalar_at<std::string>(r) = std::move(flat[static_cast<std::size_t>(base)]);
        else if (info.shape.kind() == DataKind::kVector)
            for (Index j = 0; j < info.shape[0]; ++j)
                out_ds->vector_at<std::string>(r)(j) = std::move(flat[static_cast<std::size_t>(base + j)]);
        else
            for (Index i = 0; i < info.shape[0]; ++i)
                for (Index j = 0; j < info.shape[1]; ++j)
                    out_ds->matrix_at<std::string>(r)(i, j) = std::move(flat[static_cast<std::size_t>(base + i * info.shape[1] + j)]);
    }
    // DataArray output: preserve metadata from first DataArray operand
    const DataArray* tmpl = nullptr;
    for (size_t i = 0; i < ops.size(); ++i)
        if (ops[i].is_data_array()) { tmpl = &ops[i].as_data_array(); break; }
    if (tmpl)
        return MakeArrayFromFlat(std::move(out_ds), *tmpl);
    return Value(DataArray::CreateIndependent(std::move(*out_ds)));
}

Value ExecuteMatrix(const ExecContextInfo& info,
                     const std::vector<Value>& ops) {
    if (ops.empty())
        throw std::invalid_argument("concat: empty input");

    if (info.dtype == DataType::kString)
        return ExecMatrixString(info, ops);

    switch (info.dtype) {
        case DataType::kComplex:
            return ExecMatrixT<std::complex<double>>(info, ops);
        case DataType::kReal:
            return ExecMatrixT<double>(info, ops);
        case DataType::kInteger:
            return ExecMatrixT<int>(info, ops);
        default:
            throw std::invalid_argument("concat: unsupported dtype");
    }
}

// =========================================================================
//  ExecuteSweep ([...] sweep generator) -- collect operands into DataArray
// =========================================================================
//
//  RowBroadcastPlan handles row broadcast. ShapeBroadcastPlan handles cell
//  broadcast (Scalar ↔ Vector etc.).

template <typename T>
static Value ExecSweepT(const ExecContextInfo& info,
                         const std::vector<Value>& ops) {
    std::vector<DataShape> op_shapes;
    std::vector<Index> row_counts;
    for (size_t i = 0; i < ops.size(); ++i) {
        op_shapes.push_back(ops[i].data_shape());
        row_counts.push_back(ops[i].rows());
    }

    ShapeBroadcastPlan shape_plan = ShapeBroadcastPlan::Make(op_shapes, info.shape);
    RowBroadcastPlan   row_plan   = RowBroadcastPlan::Compute(row_counts);

    std::vector<FlatInput<T>> inputs;
    for (size_t i = 0; i < ops.size(); ++i)
        inputs.push_back(FlatInput<T>::Acquire(ops[i]));

    Index cell_elems = shape_plan.result_elements;
    Index result_rows = info.rows;

    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataTypeOf<T>::tag, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(result_rows));
    T* out = out_ds->mutable_contiguous_data<T>();

    Index out_row = 0;
    for (size_t k = 0; k < ops.size(); ++k) {
        Index n_rows = ops[k].rows();
        for (Index local_r = 0; local_r < n_rows; ++local_r, ++out_row) {
            Index op_row = (ops[k].is_measurement())
                           ? 0
                           : (row_plan.broadcast[k] ? 0 : local_r);
            const T* src = inputs[k].ptr + op_row * inputs[k].stride;

            if (inputs[k].stride == cell_elems) {
                for (Index j = 0; j < cell_elems; ++j)
                    out[out_row * cell_elems + j] = src[j];
            } else {
                for (Index j = 0; j < cell_elems; ++j) {
                    Index sj = shape_plan.MapFlatIndex(j, static_cast<int>(k));
                    out[out_row * cell_elems + j] = src[sj];
                }
            }
        }
    }

    return Value(DataArray::CreateIndependent(std::move(*out_ds)));
}

// -- String path for Sweep ---

static Value ExecSweepString(const ExecContextInfo& info,
                              const std::vector<Value>& ops) {
    std::vector<DataShape> op_shapes;
    std::vector<Index> row_counts;
    for (size_t i = 0; i < ops.size(); ++i) {
        op_shapes.push_back(ops[i].data_shape());
        row_counts.push_back(ops[i].rows());
    }

    ShapeBroadcastPlan shape_plan = ShapeBroadcastPlan::Make(op_shapes, info.shape);
    RowBroadcastPlan   row_plan   = RowBroadcastPlan::Compute(row_counts);

    Index cell_elems = shape_plan.result_elements;
    Index result_rows = info.rows;
    Index total = result_rows * cell_elems;

    std::vector<std::string> flat(static_cast<std::size_t>(total));

    Index out_row = 0;
    for (size_t k = 0; k < ops.size(); ++k) {
        Index n_rows = ops[k].rows();
        for (Index local_r = 0; local_r < n_rows; ++local_r, ++out_row) {
            Index op_row = row_plan.broadcast[k] ? 0 : local_r;
            Index base = out_row * cell_elems;

            if (ops[k].is_measurement()) {
                const Measurement& m = ops[k].as_measurement();
                DataKind dk = m.data_kind();
                if (dk == DataKind::kScalar) {
                    std::string s = m.as_scalar<std::string>();
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = s;
                } else if (dk == DataKind::kVector) {
                    auto vec = m.as_vector<std::string>();
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = vec(shape_plan.MapFlatIndex(j, static_cast<int>(k)));
                } else {
                    auto mat = m.as_matrix<std::string>();
                    Index cols = m.shape()[1];
                    for (Index j = 0; j < cell_elems; ++j) {
                        Index sj = shape_plan.MapFlatIndex(j, static_cast<int>(k));
                        flat[static_cast<std::size_t>(base + j)] = mat(sj / cols, sj % cols);
                    }
                }
            } else {
                const DataSeries& ds = ops[k].as_data_array().data();
                DataKind dk = ds.data_kind();
                if (dk == DataKind::kScalar) {
                    std::string s = ds.scalar_at<std::string>(op_row);
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = s;
                } else if (dk == DataKind::kVector) {
                    auto vec = ds.vector_at<std::string>(op_row);
                    for (Index j = 0; j < cell_elems; ++j)
                        flat[static_cast<std::size_t>(base + j)] = vec(shape_plan.MapFlatIndex(j, static_cast<int>(k)));
                } else {
                    auto mat = ds.matrix_at<std::string>(op_row);
                    Index cols = ds.data_shape()[1];
                    for (Index j = 0; j < cell_elems; ++j) {
                        Index sj = shape_plan.MapFlatIndex(j, static_cast<int>(k));
                        flat[static_cast<std::size_t>(base + j)] = mat(sj / cols, sj % cols);
                    }
                }
            }
        }
    }

    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataType::kString, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(result_rows));
    for (Index r = 0; r < result_rows; ++r) {
        Index base = r * cell_elems;
        if (info.shape.kind() == DataKind::kScalar)
            out_ds->scalar_at<std::string>(r) = std::move(flat[static_cast<std::size_t>(base)]);
        else if (info.shape.kind() == DataKind::kVector)
            for (Index j = 0; j < info.shape[0]; ++j)
                out_ds->vector_at<std::string>(r)(j) = std::move(flat[static_cast<std::size_t>(base + j)]);
        else
            for (Index i = 0; i < info.shape[0]; ++i)
                for (Index j = 0; j < info.shape[1]; ++j)
                    out_ds->matrix_at<std::string>(r)(i, j) = std::move(flat[static_cast<std::size_t>(base + i * info.shape[1] + j)]);
    }
    return Value(DataArray::CreateIndependent(std::move(*out_ds)));
}

Value ExecuteSweep(const ExecContextInfo& info,
                    const std::vector<Value>& ops) {
    if (ops.empty())
        throw std::invalid_argument("sweep: empty input");

    if (info.dtype == DataType::kString)
        return ExecSweepString(info, ops);

    switch (info.dtype) {
        case DataType::kComplex:
            return ExecSweepT<std::complex<double>>(info, ops);
        case DataType::kReal:
            return ExecSweepT<double>(info, ops);
        case DataType::kInteger:
            return ExecSweepT<int>(info, ops);
        default:
            throw std::invalid_argument("sweep: unsupported dtype");
    }
}

// =========================================================================
//  ExecUnaryLoop -- core flat-buffer loop for single operand
// =========================================================================

template <typename T>
void ExecUnaryLoop(Index rows,
                    const ShapeBroadcastPlan& shape_plan,
                    const T* ptr, Index stride,
                    T* out,
                    UnaryOp<T> op)
{
    Index out_stride = shape_plan.result_elements;

    for (Index i = 0; i < rows; ++i) {
        Index i_off = i * stride;
        Index o_off = i * out_stride;

        for (Index j = 0; j < shape_plan.result_elements; ++j) {
            out[o_off + j] = op(ptr[i_off + j]);
        }
    }
}

// =========================================================================
//  ExecUnaryT -- unary entry point (reuses FlatInput, output helpers)
// =========================================================================

template <typename T>
Value ExecUnaryT(const ExecContextInfo& info,
                  const std::vector<Value>& ops,
                  UnaryOp<T> op)
{
    bool is_meas = ops[0].is_measurement();

    DataShape op_shape = ops[0].data_shape();
    ShapeBroadcastPlan shape_plan = ShapeBroadcastPlan::Make({op_shape}, info.shape);

    auto in = FlatInput<T>::Acquire(ops[0]);
    const T* ptr    = in.ptr;
    Index    stride = in.stride;

    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataTypeOf<T>::tag, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(info.rows));
    T* out = out_ds->mutable_contiguous_data<T>();

    ExecUnaryLoop(info.rows, shape_plan, ptr, stride, out, op);

    if (is_meas) {
        return Value(MakeMeasFromFlat(out, info.shape, info.unit));
    } else {
        const DataArray& src = ops[0].as_data_array();
        return MakeArrayFromFlat(std::move(out_ds), src);
    }
}

// =========================================================================
//  ExecuteUnaryNegate -- unary minus
// =========================================================================

Value ExecuteUnaryNegate(const ExecContextInfo& info,
                          const std::vector<Value>& ops) {
    switch (info.dtype) {
        case DataType::kComplex:
            return ExecUnaryT<std::complex<double>>(info, ops, op_negate);
        case DataType::kReal:
            return ExecUnaryT<double>(info, ops, op_negate);
        case DataType::kInteger:
            return ExecUnaryT<int>(info, ops, op_negate);
        default:
            throw std::invalid_argument("unsupported dtype for negation");
    }
}

// =========================================================================
//  ExecuteUnaryNot -- logical NOT (!/NOT)
// =========================================================================

Value ExecuteUnaryNot(const ExecContextInfo& info,
                       const std::vector<Value>& ops) {
    // Logical NOT: first convert to int via as_logical() (non-zero→1),
    // then apply NOT.  Scalar Meas → upgrade to Boolean.

    Value v;
    if (ops[0].is_measurement()) {
        const Measurement& m = ops[0].as_measurement();
        if (m.data_type() == DataType::kBoolean) {
            v = Value(Measurement(static_cast<int>(m.as_scalar<bool>() ? 1 : 0)));
        } else {
            DataSeries ds(m.data_kind(), m.data_type(), m.shape());
            ds.append(m);
            auto logical_ds = std::unique_ptr<DataSeries>(new DataSeries(ds.as_logical()));
            v = Value(MakeMeasFromFlat(logical_ds->contiguous_data<int>(),
                        m.shape(), m.unit()));
        }
    } else {
        auto logical_ds = std::unique_ptr<DataSeries>(new DataSeries(ops[0].as_data_array().data().as_logical()));
        v = MakeArrayFromFlat(std::move(logical_ds), ops[0].as_data_array());
    }

    Value result = ExecUnaryT<int>(info, {v}, op_not<int>);

    if (ops[0].is_measurement() && info.shape.kind() == DataKind::kScalar) {
        return Value(Measurement::Boolean(result.as_measurement().as_scalar<int>() != 0));
    }
    return result;
}

// =========================================================================
//  ExecuteUnaryBitNot -- bitwise NOT (~)
// =========================================================================

Value ExecuteUnaryBitNot(const ExecContextInfo& info,
                          const std::vector<Value>& ops) {
    return ExecUnaryT<int>(info, ops, op_bitnot<int>);
}

// =========================================================================
//  ComputeInverse -- Eigen inverse, throws if singular
// =========================================================================

template <typename T>
static Mat<T> ComputeInverse(const T* B, Index n) {
    MatConstMap<T> Bmap(B, n, n);
    Eigen::FullPivLU<Mat<T>> lu(Bmap);

    if (!lu.isInvertible())
        throw std::invalid_argument(
            "RHS matrix is singular; division undefined");

    return lu.inverse();
}

// =========================================================================
//  ExecBinaryMatMulT -- matrix multiplication with row broadcast
// =========================================================================

template <typename T>
Value ExecBinaryMatMulT(const ExecContextInfo& info,
                         const std::vector<Value>& ops)
{
    bool l_meas = ops[0].is_measurement();
    bool r_meas = ops[1].is_measurement();

    auto rcA = EffectiveRC(ops[0].data_shape());
    auto rcB = EffectiveRC(ops[1].data_shape());
    Index rA = rcA.first, cA = rcA.second;
    Index rB = rcB.first, cB = rcB.second;
    Index rC = rA, cC = cB;

    Index l_rows = ops[0].rows();
    Index r_rows = ops[1].rows();
    RowBroadcastPlan row_plan = RowBroadcastPlan::Compute({l_rows, r_rows});

    auto l_in = FlatInput<T>::Acquire(ops[0]);
    auto r_in = FlatInput<T>::Acquire(ops[1]);
    const T* l_ptr    = l_in.ptr;
    const T* r_ptr    = r_in.ptr;
    Index    l_stride = l_in.stride;
    Index    r_stride = r_in.stride;

    const DataArray* out_src = SelectOutputSource(l_meas, r_meas, ops);

    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataTypeOf<T>::tag, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(info.rows));
    T* out = out_ds->mutable_contiguous_data<T>();

    Index out_stride = rC * cC;

    for (Index i = 0; i < info.rows; ++i) {
        Index l_off = (row_plan.broadcast[0] ? 0 : i) * l_stride;
        Index r_off = (row_plan.broadcast[1] ? 0 : i) * r_stride;

        MatConstMap<T> A(l_ptr + l_off, rA, cA);
        MatConstMap<T> B(r_ptr + r_off, rB, cB);
        MatMap<T> C(out + i * out_stride, rC, cC);
        C.noalias() = A * B;
    }

    if (l_meas && r_meas) {
        return Value(MakeMeasFromFlat(out, info.shape, info.unit));
    } else {
        return MakeArrayFromFlat(std::move(out_ds), *out_src);
    }
}

// =========================================================================
//  ExecBinaryDivT -- matrix division kernel (A x inv(B)) with row broadcast
// =========================================================================

template <typename T>
Value ExecBinaryDivT(const ExecContextInfo& info,
                      const std::vector<Value>& ops)
{
    bool l_meas = ops[0].is_measurement();
    bool r_meas = ops[1].is_measurement();

    auto rcA = EffectiveRC(ops[0].data_shape());
    auto rcB = EffectiveRC(ops[1].data_shape());
    Index rA = rcA.first, cA = rcA.second;
    Index rB = rcB.first, cB = rcB.second;
    // inv(B): effective shape (cB, rB), so A x inv(B) -> (rA, rB)
    Index rC = rA, cC = rB;

    Index l_rows = ops[0].rows();
    Index r_rows = ops[1].rows();
    RowBroadcastPlan row_plan = RowBroadcastPlan::Compute({l_rows, r_rows});

    auto l_in = FlatInput<T>::Acquire(ops[0]);
    auto r_in = FlatInput<T>::Acquire(ops[1]);
    const T* l_ptr    = l_in.ptr;
    const T* r_ptr    = r_in.ptr;
    Index    l_stride = l_in.stride;
    Index    r_stride = r_in.stride;

    const DataArray* out_src = SelectOutputSource(l_meas, r_meas, ops);

    auto out_ds = std::unique_ptr<DataSeries>(
        new DataSeries(info.shape.kind(), DataTypeOf<T>::tag, info.shape));
    out_ds->set_unit(info.unit);
    out_ds->resize(static_cast<std::size_t>(info.rows));
    T* out = out_ds->mutable_contiguous_data<T>();

    Index out_stride = rC * cC;

    for (Index i = 0; i < info.rows; ++i) {
        Index l_off = (row_plan.broadcast[0] ? 0 : i) * l_stride;
        Index r_off = (row_plan.broadcast[1] ? 0 : i) * r_stride;

        // Compute inverse of B, then A x inv(B)
        Mat<T> invB = ComputeInverse(r_ptr + r_off, rB);

        MatConstMap<T> A(l_ptr + l_off, rA, cA);
        MatMap<T> C(out + i * out_stride, rC, cC);
        C.noalias() = A * invB;
    }

    if (l_meas && r_meas) {
        return Value(MakeMeasFromFlat(out, info.shape, info.unit));
    } else {
        return MakeArrayFromFlat(std::move(out_ds), *out_src);
    }
}

// =========================================================================
//  ExecuteBinaryMul -- dispatches scalar (element-wise) vs matrix multiply
// =========================================================================

Value ExecuteBinaryMul(const ExecContextInfo& info,
                        const std::vector<Value>& ops) {
    // Scalar path: either operand is scalar -> element-wise broadcast
    if (ops[0].data_shape().kind() == DataKind::kScalar ||
        ops[1].data_shape().kind() == DataKind::kScalar) {
        return ExecuteBinaryArith(info, ops);
    }

    // Matrix multiplication path
    switch (info.dtype) {
        case DataType::kComplex:
            return ExecBinaryMatMulT<std::complex<double>>(info, ops);
        case DataType::kReal:
            return ExecBinaryMatMulT<double>(info, ops);
        case DataType::kInteger:
            return ExecBinaryMatMulT<int>(info, ops);
        default:
            throw std::invalid_argument("unsupported dtype for matmul");
    }
}

// =========================================================================
//  ExecuteBinaryDiv -- dispatches scalar (element-wise) vs A x inv(B)
// =========================================================================

Value ExecuteBinaryDiv(const ExecContextInfo& info,
                        const std::vector<Value>& ops) {
    // Scalar path: RHS is scalar -> element-wise broadcast
    if (ops[1].data_shape().kind() == DataKind::kScalar) {
        return ExecuteBinaryArith(info, ops);
    }

    // Matrix division path: A x inv(B)
    // (dtype is always >=Real; DeriveDtypeDiv promotes int->real)
    switch (info.dtype) {
        case DataType::kComplex:
            return ExecBinaryDivT<std::complex<double>>(info, ops);
        case DataType::kReal:
            return ExecBinaryDivT<double>(info, ops);
        default:
            throw std::invalid_argument("unsupported dtype for div");
    }
}

// =========================================================================
//  Predefined OpTraits
// =========================================================================

// ---- binary arithmetic -----------------------------------------------------

const OpTraits kOpAdd = {
    OpCategory::kAdd, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitSameDim, ExecuteBinaryArith
};

const OpTraits kOpSub = {
    OpCategory::kSub, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitSameDim, ExecuteBinaryArith
};

const OpTraits kOpMul = {
    OpCategory::kMul, 2, DeriveShapeMul, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitMul, ExecuteBinaryMul
};

const OpTraits kOpDiv = {
    OpCategory::kDiv, 2, DeriveShapeDiv, DeriveRowsBroadcast,
    DeriveDtypeDiv, DeriveUnitDiv, ExecuteBinaryDiv
};

const OpTraits kOpMod = {
    OpCategory::kMod, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeMod, DeriveUnitSameDim, ExecuteBinaryArith
};

const OpTraits kOpPow = {
    OpCategory::kPow, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePow, DeriveUnitFirst, ExecuteBinaryArith
};

// ---- binary comparison -----------------------------------------------------

const OpTraits kOpEq = {
    OpCategory::kEq, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeCmp, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpNeq = {
    OpCategory::kNeq, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeCmp, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpLt = {
    OpCategory::kLt, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeCmp, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpGt = {
    OpCategory::kGt, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeCmp, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpLe = {
    OpCategory::kLe, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeCmp, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpGe = {
    OpCategory::kGe, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeCmp, DeriveUnitDimless, ExecuteBinaryCmp
};

// ---- binary logical --------------------------------------------------------

const OpTraits kOpAnd = {
    OpCategory::kAnd, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteBinaryLogical
};

const OpTraits kOpOr = {
    OpCategory::kOr, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteBinaryLogical
};

// ---- binary bitwise ----------------------------------------------------

const OpTraits kOpBitAnd = {
    OpCategory::kBitAnd, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeBitwise, DeriveUnitDimless, ExecuteBinaryBitwise
};

const OpTraits kOpBitOr = {
    OpCategory::kBitOr, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeBitwise, DeriveUnitDimless, ExecuteBinaryBitwise
};

const OpTraits kOpBitXor = {
    OpCategory::kBitXor, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeBitwise, DeriveUnitDimless, ExecuteBinaryBitwise
};

// ---- binary shift ------------------------------------------------------

const OpTraits kOpShl = {
    OpCategory::kShl, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeBitwise, DeriveUnitDimless, ExecuteBinaryShift
};

const OpTraits kOpShr = {
    OpCategory::kShr, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeBitwise, DeriveUnitDimless, ExecuteBinaryShift
};

// ---- unary ----------------------------------------------------------------

const OpTraits kOpNegate = {
    OpCategory::kNegate, 1, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, ExecuteUnaryNegate
};

const OpTraits kOpNot = {
    OpCategory::kNot, 1, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteUnaryNot
};

const OpTraits kOpBitNot = {
    OpCategory::kBitNot, 1, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeBitwise, DeriveUnitDimless, ExecuteUnaryBitNot
};

// ---- ternary (TODO: execute) -----------------------------------------------

const OpTraits kOpConditional = {
    OpCategory::kConditional, 3, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, nullptr
};

// ---- variadic (TODO: execute) ----------------------------------------------

const OpTraits kOpSweep = {
    OpCategory::kSweep, Arity::kVariadic, DeriveShapeBroadcast,
    DeriveRowsSum, DeriveDtypeMerge, DeriveUnitSameDim, ExecuteSweep
};

const OpTraits kOpMatrix = {
    OpCategory::kMatrix, Arity::kVariadic, DeriveShapeMatrix,
    DeriveRowsBroadcast, DeriveDtypeMerge, DeriveUnitSameDim, ExecuteMatrix
};

// =========================================================================
//  Public API wrappers
// =========================================================================

Value OperationAdd(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpAdd); }
Value OperationSub(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpSub); }
Value OperationMul(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpMul); }
Value OperationDiv(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpDiv); }
Value OperationMod(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpMod); }
Value OperationPow(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpPow); }

Value OperationEq(const Value& lhs, const Value& rhs)    { return Operate({lhs, rhs}, kOpEq); }
Value OperationNeq(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpNeq); }
Value OperationLt(const Value& lhs, const Value& rhs)    { return Operate({lhs, rhs}, kOpLt); }
Value OperationGt(const Value& lhs, const Value& rhs)    { return Operate({lhs, rhs}, kOpGt); }
Value OperationLe(const Value& lhs, const Value& rhs)    { return Operate({lhs, rhs}, kOpLe); }
Value OperationGe(const Value& lhs, const Value& rhs)    { return Operate({lhs, rhs}, kOpGe); }

Value OperationAnd(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpAnd); }
Value OperationOr(const Value& lhs, const Value& rhs)    { return Operate({lhs, rhs}, kOpOr); }

Value OperationBitAnd(const Value& lhs, const Value& rhs){ return Operate({lhs, rhs}, kOpBitAnd); }
Value OperationBitOr(const Value& lhs, const Value& rhs) { return Operate({lhs, rhs}, kOpBitOr); }
Value OperationBitXor(const Value& lhs, const Value& rhs){ return Operate({lhs, rhs}, kOpBitXor); }

Value OperationShl(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpShl); }
Value OperationShr(const Value& lhs, const Value& rhs)   { return Operate({lhs, rhs}, kOpShr); }

Value OperationNegate(const Value& v)                    { return Operate({v}, kOpNegate); }
Value OperationNot(const Value& v)                       { return Operate({v}, kOpNot); }
Value OperationBitNot(const Value& v)                    { return Operate({v}, kOpBitNot); }

Value OperationMatrix(const std::vector<Value>& ops)     { return Operate(ops, kOpMatrix); }
Value OperationSweep(const std::vector<Value>& ops)      { return Operate(ops, kOpSweep); }

}  // namespace xdataset
