#ifndef XDATASET_OPERATION_H
#define XDATASET_OPERATION_H

#include "value.h"

#include <vector>

namespace xdataset {

// =========================================================================
// OpCategory -- operator category enumeration
// =========================================================================
enum class OpCategory {
    // arithmetic
    kAdd, kSub, kMul, kDiv, kMod,
    // comparison
    kEq, kNeq, kLt, kGt, kLe, kGe,
    // logical
    kAnd, kOr,
    // bitwise
    kBitAnd, kBitOr, kBitXor,
    // shift
    kShl, kShr,
    // power
    kPow,
    // unary
    kNegate, kNot, kBitNot,
    // ternary
    kConditional,
    // variadic
    kConcat, kCombine
};

// =========================================================================
// DataShape -- defined in xdataset_predefine.h.  Key members:
//
//   DataShape{}     → kScalar     ds.kind() / ds.element_count()
//   DataShape{w}    → kVector
//   DataShape{r,c}  → kMatrix
// =========================================================================

// =========================================================================
// RowBroadcastPlan -- row 维度的广播计划
// =========================================================================
//
// 规则：size=1 的 operand 广播到 max size；>1 的必须全部相等。

struct RowBroadcastPlan {
    Index              result_size;
    std::vector<bool>  broadcast;     // per-operand: true=始终取索引0

    /// 核心算法：给定 N 个 operand 在 row 轴上的大小，计算广播计划。
    /// 例: {1, 100}  → result_size=100, broadcast={true, false}
    ///     {100,100} → result_size=100, broadcast={false,false}
    static RowBroadcastPlan Compute(const std::vector<Index>& sizes);
};

// =========================================================================
// OperandBroadcastShapeInfo -- 单个 operand 的 cell 内形状和广播信息
// =========================================================================
struct OperandBroadcastShapeInfo {
    Index elements;        // cell 内元素总数
    Index cols;            // scalar=1, vector=shape[0] (单行), matrix=shape[1]
    bool  broadcast_row;   // row 维度是否广播
    bool  broadcast_col;   // col 维度是否广播
};

// =========================================================================
// ShapeBroadcastPlan -- cell 内元素广播计划（支持 Scalar/Vector/Matrix/Vec×Mat）
// =========================================================================
//
// Vector 视为单行 (row=1, col=w)，与 Matrix 的 col 维对齐。
// 每个维度独立广播：1→N，>1 的必须相等。
//
//   Vector [w] × Matrix [r, w]  → 广播行 (1→r)，列对齐 (w==w)
//   Vector [w] × Matrix [r, c]  → w!=c 时抛异常
//   Vector [1] × Matrix [r, c]  → 全广播 (1→r, 1→c)

struct ShapeBroadcastPlan {
    DataShape          result_shape;
    Index              result_elements;
    Index              result_cols;      // 1 for scalar, shape[0] for vector, shape[1] for matrix

    std::vector<OperandBroadcastShapeInfo> ops;

    static ShapeBroadcastPlan Make(const std::vector<DataShape>& operand_shapes,
                                    const DataShape& result);

    Index MapFlatIndex(Index result_flat, int k) const;
};

// =========================================================================
// ExecContextInfo -- all derived metadata passed to execute
// =========================================================================
struct ExecContextInfo {
    OpCategory         op;
    Index              rows;
    DataShape          shape;
    DataType           dtype;
    Unit               unit;
};

// =========================================================================
// Element-wise operation type (for unified T* loop)
// =========================================================================
template <typename T>
using ElemOp = T (*)(T, T);

// =========================================================================
// Callback type aliases (C++11)
// =========================================================================

/// Shape derivation — returns result DataShape (kind via ds.kind())
typedef DataShape (*DeriveShapeFunc)(const std::vector<DataShape>& operand_shapes,
                                      OpCategory category);

/// Type derivation
typedef DataType (*DeriveDtypeFunc)(const std::vector<DataType>& dtypes,
                                     OpCategory category);

/// Unit derivation
typedef Unit (*DeriveUnitFunc)(const std::vector<Unit>& units,
                                OpCategory category);

/// Row derivation — returns result row count
typedef Index (*DeriveRowsFunc)(const std::vector<Index>& rows,
                                 OpCategory category);

/// Execute: receives ExecContextInfo + operands, returns Value
typedef Value (*ExecuteFunc)(const ExecContextInfo& info,
                              const std::vector<Value>& ops);

// =========================================================================
// Arity -- operand count constraint
// =========================================================================

/// Positive = fixed count, kVariadic (= -1) = unlimited.
/// Operate checks operands.size() before invoking callbacks.
enum Arity : Index {
    kVariadic = -1
};

// =========================================================================
// OpTraits -- complete description of one operation
// =========================================================================
//
// Every field is a function pointer. Errors are thrown directly inside
// callbacks. There is no separate validate callback -- validation logic
// is distributed into the derive callbacks.

struct OpTraits {
    OpCategory      op;

    /// Required operand count. kVariadic (= -1) means unlimited.
    Index           arity;

    DeriveShapeFunc derive_shape;
    DeriveRowsFunc  derive_rows;
    DeriveDtypeFunc derive_dtype;
    DeriveUnitFunc  derive_unit;

    ExecuteFunc     execute;
};

// =========================================================================
// Public API
// =========================================================================

/// Derive + execute. Only extracts parameters and forwards to callbacks;
/// does not perform any type promotion itself.
XDATASET_API Value Operate(const std::vector<Value>& operands,
                            const OpTraits& traits);

// =========================================================================
// Predefined derive callbacks (for building OpTraits)
// =========================================================================

/// ── derive_shape ───────────────────────────────────────────────────
XDATASET_API DataShape DeriveShapeBroadcast(const std::vector<DataShape>& operand_shapes,
                                             OpCategory category);
XDATASET_API DataShape DeriveShapeConcat(const std::vector<DataShape>& operand_shapes,
                                          OpCategory category);

/// ── derive_rows ────────────────────────────────────────────────────
XDATASET_API Index DeriveRowsBroadcast(const std::vector<Index>& rows,
                                        OpCategory category);

/// ── derive_dtype ───────────────────────────────────────────────────
XDATASET_API DataType DeriveDtypePromote(const std::vector<DataType>& dtypes,
                                          OpCategory category);
XDATASET_API DataType DeriveDtypeDiv(const std::vector<DataType>& dtypes,
                                      OpCategory category);
XDATASET_API DataType DeriveDtypeForceInt(const std::vector<DataType>& dtypes,
                                           OpCategory category);

/// ── derive_unit ────────────────────────────────────────────────────
XDATASET_API Unit DeriveUnitSameDim(const std::vector<Unit>& units,
                                 OpCategory category);
XDATASET_API Unit DeriveUnitMul(const std::vector<Unit>& units,
                                 OpCategory category);
XDATASET_API Unit DeriveUnitDiv(const std::vector<Unit>& units,
                                 OpCategory category);
XDATASET_API Unit DeriveUnitDimless(const std::vector<Unit>& units,
                                     OpCategory category);
XDATASET_API Unit DeriveUnitFirst(const std::vector<Unit>& units,
                                   OpCategory category);

// =========================================================================
// Predefined execute callbacks
// =========================================================================

XDATASET_API Value ExecuteBinaryArith(const ExecContextInfo& info,
                                       const std::vector<Value>& ops);
XDATASET_API Value ExecuteBinaryCmp(const ExecContextInfo& info,
                                     const std::vector<Value>& ops);
XDATASET_API Value ExecuteBinaryLogical(const ExecContextInfo& info,
                                         const std::vector<Value>& ops);

// =========================================================================
// Predefined OpTraits instances
// =========================================================================
extern XDATASET_API const OpTraits kOpAdd;
extern XDATASET_API const OpTraits kOpSub;
extern XDATASET_API const OpTraits kOpMul;
extern XDATASET_API const OpTraits kOpDiv;
extern XDATASET_API const OpTraits kOpEq;
extern XDATASET_API const OpTraits kOpNeq;
extern XDATASET_API const OpTraits kOpLt;
extern XDATASET_API const OpTraits kOpGt;
extern XDATASET_API const OpTraits kOpLe;
extern XDATASET_API const OpTraits kOpGe;
extern XDATASET_API const OpTraits kOpAnd;
extern XDATASET_API const OpTraits kOpOr;
extern XDATASET_API const OpTraits kOpBitAnd;
extern XDATASET_API const OpTraits kOpBitOr;
extern XDATASET_API const OpTraits kOpBitXor;
extern XDATASET_API const OpTraits kOpShl;
extern XDATASET_API const OpTraits kOpShr;
extern XDATASET_API const OpTraits kOpNegate;
extern XDATASET_API const OpTraits kOpNot;
extern XDATASET_API const OpTraits kOpBitNot;
extern XDATASET_API const OpTraits kOpConditional;
extern XDATASET_API const OpTraits kOpMod;
extern XDATASET_API const OpTraits kOpCombine;
extern XDATASET_API const OpTraits kOpConcat;

}  // namespace xdataset

#endif  // XDATASET_OPERATION_H
