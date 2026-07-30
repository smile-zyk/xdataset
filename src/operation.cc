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

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace xdataset {

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

    // Extract per-operand metadata
    std::vector<DataShape> operand_shapes;
    std::vector<Index>     row_counts;
    std::vector<DataType>  dtypes;
    std::vector<Unit>      units;

    for (size_t i = 0; i < operands.size(); ++i) {
        operand_shapes.push_back(operands[i].shape());
        row_counts.push_back(operands[i].rows());
        dtypes.push_back(operands[i].data_type());
        units.push_back(operands[i].unit());
    }

    // Derive result metadata from operands
    DataShape shape       = traits.derive_shape(operand_shapes, traits.op);
    Index     rows        = traits.derive_rows(row_counts, traits.op);
    DataType  dtype       = traits.derive_dtype(dtypes, traits.op);
    Unit      unit        = traits.derive_unit(units, traits.op);

    // Pack context
    ExecContextInfo info;
    info.op    = traits.op;
    info.rows  = rows;
    info.shape = shape;
    info.dtype = dtype;
    info.unit  = unit;

    return traits.execute(info, operands);
}

// =========================================================================
//  Derive callbacks
// =========================================================================

// -- DeriveShapeBroadcast --

DataShape DeriveShapeBroadcast(const std::vector<DataShape>& operand_shapes,
                                OpCategory /*category*/) {
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

// -- DeriveShapeConcat --

DataShape DeriveShapeConcat(const std::vector<DataShape>& operand_shapes,
                             OpCategory /*category*/) {
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

// -- DeriveRowsBroadcast --

Index DeriveRowsBroadcast(const std::vector<Index>& rows,
                          OpCategory /*category*/) {
    return RowBroadcastPlan::Compute(rows).result_size;
}

// -- DeriveDtypePromote --

DataType DeriveDtypePromote(const std::vector<DataType>& dtypes,
                             OpCategory /*category*/) {
    DataType res = DataType::kInteger;
    for (size_t i = 0; i < dtypes.size(); ++i) {
        if (dtypes[i] == DataType::kComplex)
            res = DataType::kComplex;
        else if (dtypes[i] == DataType::kReal && res != DataType::kComplex)
            res = DataType::kReal;
    }
    return res;
}

// -- DeriveDtypeDiv --

DataType DeriveDtypeDiv(const std::vector<DataType>& dtypes,
                         OpCategory /*category*/) {
    DataType res = DeriveDtypePromote(dtypes, OpCategory::kAdd);
    if (res == DataType::kInteger) res = DataType::kReal;
    return res;
}

// -- DeriveDtypeForceInt --

DataType DeriveDtypeForceInt(const std::vector<DataType>& /*dtypes*/,
                              OpCategory /*category*/) {
    return DataType::kInteger;
}

// -- DeriveUnitSameDim --

Unit DeriveUnitSameDim(const std::vector<Unit>& units, OpCategory /*category*/) {
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

Unit DeriveUnitMul(const std::vector<Unit>& units, OpCategory /*category*/) {
    Unit res = units[0];
    for (size_t i = 1; i < units.size(); ++i)
        res = res * units[i];
    return res;
}

// -- DeriveUnitDiv --

Unit DeriveUnitDiv(const std::vector<Unit>& units, OpCategory /*category*/) {
    Unit res = units[0];
    for (size_t i = 1; i < units.size(); ++i)
        res = res / units[i];
    return res;
}

// -- DeriveUnitDimless --

Unit DeriveUnitDimless(const std::vector<Unit>& /*units*/,
                        OpCategory /*category*/) {
    return Unit();
}

// -- DeriveUnitFirst --

Unit DeriveUnitFirst(const std::vector<Unit>& units,
                      OpCategory /*category*/) {
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

template <typename T> inline T op_eq(T a, T b) {
    return static_cast<T>(a == b ? 1 : 0);
}
template <typename T> inline T op_ne(T a, T b) {
    return static_cast<T>(a != b ? 1 : 0);
}
template <typename T> inline T op_lt(T a, T b) {
    return static_cast<T>(a <  b ? 1 : 0);
}
template <typename T> inline T op_gt(T a, T b) {
    return static_cast<T>(a >  b ? 1 : 0);
}
template <typename T> inline T op_le(T a, T b) {
    return static_cast<T>(a <= b ? 1 : 0);
}
template <typename T> inline T op_ge(T a, T b) {
    return static_cast<T>(a >= b ? 1 : 0);
}

template <typename T> inline T op_and(T a, T b) {
    return static_cast<T>(
        (static_cast<int>(a) && static_cast<int>(b)) ? 1 : 0);
}
template <typename T> inline T op_or(T a, T b) {
    return static_cast<T>(
        (static_cast<int>(a) || static_cast<int>(b)) ? 1 : 0);
}

template <typename T>
ElemOp<T> GetArithElemOp(OpCategory cat) {
    switch (cat) {
        case OpCategory::kAdd: return op_add<T>;
        case OpCategory::kSub: return op_sub<T>;
        case OpCategory::kMul: return op_mul<T>;
        case OpCategory::kDiv: return op_div<T>;
        case OpCategory::kMod: return op_mod<T>;
        default: throw std::invalid_argument("not an arithmetic op");
    }
}

template <typename T>
ElemOp<T> GetCmpElemOp(OpCategory cat) {
    switch (cat) {
        case OpCategory::kEq:  return op_eq<T>;
        case OpCategory::kNeq: return op_ne<T>;
        case OpCategory::kLt:  return op_lt<T>;
        case OpCategory::kGt:  return op_gt<T>;
        case OpCategory::kLe:  return op_le<T>;
        case OpCategory::kGe:  return op_ge<T>;
        default: throw std::invalid_argument("not a comparison op");
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

}  // anonymous namespace

// =========================================================================
//  ExecBinaryLoop -- core flat-buffer loop
// =========================================================================
//
//  Row-level and cell-level broadcast are driven by the two plans.
//  This function does not depend on ExecContextInfo.

template <typename T>
void ExecBinaryLoop(Index rows,
                     const RowBroadcastPlan&   row_plan,
                     const ShapeBroadcastPlan& shape_plan,
                     const T* l_ptr, Index l_stride,
                     const T* r_ptr, Index r_stride,
                     T* out,
                     ElemOp<T> elem_op)
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

template <typename T>
Measurement MakeMeasFromFlat(const T* data,
                              const ShapeBroadcastPlan& sp,
                              const Unit& unit) {
    DataKind dk = sp.result_shape.kind();

    if (dk == DataKind::kScalar)
        return Measurement(data[0], unit);

    if (dk == DataKind::kVector) {
        Index w = sp.result_cols;
        Eigen::Matrix<T, 1, Eigen::Dynamic> v(w);
        for (Index i = 0; i < w; ++i)
            v(i) = data[i];
        return Measurement(v, unit);
    }

    Index r = sp.result_elements / sp.result_cols;
    Index c = sp.result_cols;
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

    static FlatInput Acquire(const Value& v);
};

template <typename T>
FlatInput<T> FlatInput<T>::Acquire(const Value& v) {
    if (v.is_meas()) {
        FlatInput fi;
        const Measurement& m = v.as_meas();
        fi.owner = std::unique_ptr<DataSeries>(
            new DataSeries(m.data_kind(), m.data_type(), m.shape()));
        fi.owner->append(m);
        if (m.data_type() != DataTypeOf<T>::tag)
            fi.owner->promote_dtype(DataTypeOf<T>::tag);
        fi.ptr    = fi.owner->template contiguous_data<T>();
        fi.stride = static_cast<Index>(fi.owner->element_count());
        return fi;
    }

    // DataArray
    FlatInput fi;
    const DataSeries& src = v.as_array().data();
    if (src.data_type() == DataTypeOf<T>::tag) {
        // borrow directly — no copy
        fi.ptr    = src.contiguous_data<T>();
        fi.stride = static_cast<Index>(src.element_count());
    } else {
        // copy and promote
        fi.owner = std::unique_ptr<DataSeries>(new DataSeries(src));
        fi.owner->promote_dtype(DataTypeOf<T>::tag);
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
    bool l_meas = ops[0].is_meas();
    bool r_meas = ops[1].is_meas();

    // --- 提取 per-operand shape/rows ---
    DataShape l_shape = ops[0].shape();
    DataShape r_shape = ops[1].shape();
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

    // ====================================
    //  Step 2: 分配输出
    // ====================================
    const DataArray* out_src = NULL;
    if (!l_meas && !r_meas)
        out_src = &ops[0].as_array();
    else if (l_meas && !r_meas)
        out_src = &ops[1].as_array();
    else if (!l_meas && r_meas)
        out_src = &ops[0].as_array();

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
        return Value(MakeMeasFromFlat(out, shape_plan, info.unit));
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

Value ExecuteBinaryCmp(const ExecContextInfo& info,
                        const std::vector<Value>& ops) {
    return ExecBinaryArithT<int>(info, ops,
        GetCmpElemOp<int>(info.op));
}

Value ExecuteBinaryLogical(const ExecContextInfo& info,
                            const std::vector<Value>& ops) {
    return ExecBinaryArithT<int>(info, ops,
        GetLogicalElemOp<int>(info.op));
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
    OpCategory::kMul, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitMul, ExecuteBinaryArith
};

const OpTraits kOpDiv = {
    OpCategory::kDiv, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeDiv, DeriveUnitDiv, ExecuteBinaryArith
};

const OpTraits kOpMod = {
    OpCategory::kMod, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, ExecuteBinaryArith
};

// ---- binary comparison -----------------------------------------------------

const OpTraits kOpEq = {
    OpCategory::kEq, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpNeq = {
    OpCategory::kNeq, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpLt = {
    OpCategory::kLt, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpGt = {
    OpCategory::kGt, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpLe = {
    OpCategory::kLe, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteBinaryCmp
};

const OpTraits kOpGe = {
    OpCategory::kGe, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, ExecuteBinaryCmp
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

// ---- binary bitwise (TODO: execute) ----------------------------------------

const OpTraits kOpBitAnd = {
    OpCategory::kBitAnd, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpBitOr = {
    OpCategory::kBitOr, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpBitXor = {
    OpCategory::kBitXor, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

// ---- binary shift (TODO: execute) ------------------------------------------

const OpTraits kOpShl = {
    OpCategory::kShl, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitFirst, NULL
};

const OpTraits kOpShr = {
    OpCategory::kShr, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitFirst, NULL
};

// ---- unary (TODO: execute) -------------------------------------------------

const OpTraits kOpNegate = {
    OpCategory::kNegate, 1, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, NULL
};

const OpTraits kOpNot = {
    OpCategory::kNot, 1, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpBitNot = {
    OpCategory::kBitNot, 1, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

// ---- ternary (TODO: execute) -----------------------------------------------

const OpTraits kOpConditional = {
    OpCategory::kConditional, 3, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, NULL
};

// ---- variadic (TODO: execute) ----------------------------------------------

const OpTraits kOpCombine = {
    OpCategory::kCombine, Arity::kVariadic, DeriveShapeBroadcast,
    DeriveRowsBroadcast, DeriveDtypePromote, DeriveUnitFirst, NULL
};

const OpTraits kOpConcat = {
    OpCategory::kConcat, Arity::kVariadic, DeriveShapeConcat,
    DeriveRowsBroadcast, DeriveDtypePromote, DeriveUnitFirst, NULL
};

}  // namespace xdataset
