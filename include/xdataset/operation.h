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
// ShapeInfo -- return type of derive_shape callback
// =========================================================================
struct ShapeInfo {
    DataKind           kind;
    std::vector<Index> shape;
};

// =========================================================================
// ExecContextInfo -- all derived metadata passed to execute
// =========================================================================
struct ExecContextInfo {
    OpCategory         op;
    ShapeInfo          shape;
    Index              rows;
    DataType           dtype;
    Unit               unit;
};

// =========================================================================
// Callback type aliases (C++11)
// =========================================================================

/// Shape derivation (kind + shape combined)
typedef ShapeInfo (*DeriveShapeFunc)(const std::vector<ShapeInfo>& operand_shapes,
                                      OpCategory category);

/// Type derivation
typedef DataType (*DeriveDtypeFunc)(const std::vector<DataType>& dtypes,
                                     OpCategory category);

/// Unit derivation
typedef Unit (*DeriveUnitFunc)(const std::vector<Unit>& units,
                                OpCategory category);

/// Row derivation
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
// Operate -- unified entry point
// =========================================================================

/// Derive + execute. Only extracts parameters and forwards to callbacks;
/// does not perform any type promotion itself.
XDATASET_API Value Operate(const std::vector<Value>& operands,
                            const OpTraits& traits);

// =========================================================================
// Predefined derive callbacks (for building OpTraits)
// =========================================================================

/// ── derive_shape ───────────────────────────────────────────────────
XDATASET_API ShapeInfo DeriveShapeBroadcast(const std::vector<ShapeInfo>& operand_shapes,
                                             OpCategory category);
XDATASET_API ShapeInfo DeriveShapeConcat(const std::vector<ShapeInfo>& operand_shapes,
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
XDATASET_API Unit DeriveUnitAdd(const std::vector<Unit>& units,
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
// Predefined OpTraits instances (execute is NULL for now)
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
