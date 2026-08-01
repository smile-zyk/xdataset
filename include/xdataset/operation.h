#ifndef XDATASET_OPERATION_H
#define XDATASET_OPERATION_H

#include "value.h"

#include <vector>

namespace xdataset {

// =========================================================================
// Binary arithmetic
// =========================================================================

XDATASET_API Value OperationAdd(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationSub(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationMul(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationDiv(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationMod(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationPow(const Value& lhs, const Value& rhs);

// =========================================================================
// Unary
// =========================================================================

XDATASET_API Value OperationNegate(const Value& v);
XDATASET_API Value OperationNot(const Value& v);
XDATASET_API Value OperationBitNot(const Value& v);

// =========================================================================
// Comparison (result is Integer 0/1, dimensionless)
// =========================================================================

XDATASET_API Value OperationEq(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationNeq(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationLt(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationGt(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationLe(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationGe(const Value& lhs, const Value& rhs);

// =========================================================================
// Bitwise (Integer only, dimensionless)
// =========================================================================

XDATASET_API Value OperationBitAnd(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationBitOr(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationBitXor(const Value& lhs, const Value& rhs);

// =========================================================================
// Shift (Integer only)
// =========================================================================

XDATASET_API Value OperationShl(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationShr(const Value& lhs, const Value& rhs);

// =========================================================================
// Logical (result is Integer 0/1, dimensionless)
// =========================================================================

XDATASET_API Value OperationAnd(const Value& lhs, const Value& rhs);
XDATASET_API Value OperationOr(const Value& lhs, const Value& rhs);

// =========================================================================
// Ternary
// =========================================================================

/// Conditional(condition, true_value, false_value) — ternary operator.
/// condition is evaluated as logical (non-zero → 1, zero → 0).
/// For each element, if condition is 1 the result is taken from true_value,
/// otherwise from false_value.  Supports row broadcast and shape broadcast.
XDATASET_API Value OperationConditional(const Value& condition,
                                         const Value& true_value,
                                         const Value& false_value);
/// If(cond0, val0, cond1, val1, ..., cond_{n-1}, val_{n-1}, else_val)
/// — multi-branch if/elseif/else.  Takes 2n+1 operands (n >= 1).
/// For each element, the first branch whose condition is non-zero provides
/// the result; if no condition matches, the final else_val is used.
/// This generalizes Conditional to an arbitrary number of branches.
XDATASET_API Value OperationIf(const std::vector<Value>& operands);
// =========================================================================
// Variadic generators
// =========================================================================

/// Matrix {} — stack operands with row broadcast.
XDATASET_API Value OperationMatrix(const std::vector<Value>& operands);

/// Sweep [] — collect operands into a DataArray (one row per operand).
XDATASET_API Value OperationSweep(const std::vector<Value>& operands);

}  // namespace xdataset

#endif  // XDATASET_OPERATION_H
