// =============================================================================
//  xdataset -- operation framework
// =============================================================================
//
//  Pipeline (derive + execute), derive callbacks, and predefined OpTraits
//  instances.  Execute callbacks are NULL for now �?derivation framework only.

#include "operation.h"

#include <stdexcept>
#include <string>

namespace xdataset {

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
    std::vector<ShapeInfo> operand_shapes;
    std::vector<Index>     row_counts;
    std::vector<DataType>  dtypes;
    std::vector<Unit>      units;

    for (size_t i = 0; i < operands.size(); ++i) {
        ShapeInfo s;
        s.kind  = operands[i].data_kind();
        s.shape = operands[i].shape();
        operand_shapes.push_back(s);
        row_counts.push_back(operands[i].rows());
        dtypes.push_back(operands[i].data_type());
        units.push_back(operands[i].unit());
    }

    // Derive
    ShapeInfo ds    = traits.derive_shape(operand_shapes, traits.op);
    Index     rows  = traits.derive_rows(row_counts, traits.op);
    DataType     dtype = traits.derive_dtype(dtypes, traits.op);
    Unit         unit  = traits.derive_unit(units, traits.op);

    ExecContextInfo info;
    info.op    = traits.op;
    info.shape = ds;
    info.rows  = rows;
    info.dtype = dtype;
    info.unit  = unit;

    return traits.execute(info, operands);
}

// =========================================================================
//  Derive callbacks
// =========================================================================

// ---- derive_shape -------------------------------------------------------

ShapeInfo DeriveShapeBroadcast(const std::vector<ShapeInfo>& operand_shapes,
                                OpCategory /*category*/) {
    DataKind res_kind = DataKind::kScalar;
    const std::vector<Index>* ref_shape = NULL;

    for (size_t i = 0; i < operand_shapes.size(); ++i) {
        DataKind k = operand_shapes[i].kind;
        if (k == DataKind::kMatrix && res_kind == DataKind::kVector)
            throw std::invalid_argument("Vector x Matrix not allowed in arithmetic");
        if (k == DataKind::kVector && res_kind == DataKind::kMatrix)
            throw std::invalid_argument("Vector x Matrix not allowed in arithmetic");

        if (k == DataKind::kMatrix)
            res_kind = DataKind::kMatrix;
        else if (k == DataKind::kVector && res_kind != DataKind::kMatrix)
            res_kind = DataKind::kVector;

        if (k == DataKind::kScalar) continue;
        if (!ref_shape) { ref_shape = &operand_shapes[i].shape; continue; }
        if (*ref_shape != operand_shapes[i].shape)
            throw std::invalid_argument("shape mismatch at index " + std::to_string(i));
    }

    return {res_kind, ref_shape ? *ref_shape : std::vector<Index>{}};
}

ShapeInfo DeriveShapeConcat(const std::vector<ShapeInfo>& operand_shapes,
                             OpCategory /*category*/) {
    if (operand_shapes.empty())
        throw std::invalid_argument("concat: empty input");

    const DataKind k0 = operand_shapes[0].kind;
    const std::vector<Index>& s0 = operand_shapes[0].shape;
    for (size_t i = 1; i < operand_shapes.size(); ++i) {
        if (operand_shapes[i].kind != k0)
            throw std::invalid_argument("concat: kind mismatch at index " + std::to_string(i));
        if (operand_shapes[i].shape != s0)
            throw std::invalid_argument("concat: shape mismatch at index " + std::to_string(i));
    }

    const Index N = static_cast<Index>(operand_shapes.size());

    if (k0 == DataKind::kScalar)
        return {DataKind::kVector, {N}};
    if (k0 == DataKind::kVector)
        return {DataKind::kMatrix, {N, s0[0]}};

    throw std::invalid_argument("concat: cannot concat matrices");
}

// ---- derive_rows --------------------------------------------------------

Index DeriveRowsBroadcast(const std::vector<Index>& rows, OpCategory /*category*/) {
    Index res = 1;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] == 1) continue;
        if (res == 1) { res = rows[i]; continue; }
        if (rows[i] != res)
            throw std::invalid_argument("row count mismatch (" +
                std::to_string(res) + " vs " + std::to_string(rows[i]) + ")");
    }
    return res;
}

// ---- derive_dtype -------------------------------------------------------

DataType DeriveDtypePromote(const std::vector<DataType>& dtypes, OpCategory /*category*/) {
    DataType res = DataType::kInteger;
    for (size_t i = 0; i < dtypes.size(); ++i) {
        if (dtypes[i] == DataType::kComplex)
            res = DataType::kComplex;
        else if (dtypes[i] == DataType::kReal && res != DataType::kComplex)
            res = DataType::kReal;
    }
    return res;
}

DataType DeriveDtypeDiv(const std::vector<DataType>& dtypes, OpCategory /*category*/) {
    DataType res = DeriveDtypePromote(dtypes, OpCategory::kAdd);
    if (res == DataType::kInteger) res = DataType::kReal;
    return res;
}

DataType DeriveDtypeForceInt(const std::vector<DataType>& /*dtypes*/, OpCategory /*category*/) {
    return DataType::kInteger;
}

// ---- derive_unit ---------------------------------------------------------

Unit DeriveUnitAdd(const std::vector<Unit>& units, OpCategory /*category*/) {
    Unit res = units[0];
    for (size_t i = 1; i < units.size(); ++i) {
        if (res.same_dimension(units[i])) continue;
        if (!res.has_dimension()) { res = units[i]; continue; }
        if (!units[i].has_dimension()) continue;
        throw std::invalid_argument("unit dimension mismatch");
    }
    return res;
}

Unit DeriveUnitMul(const std::vector<Unit>& units, OpCategory /*category*/) {
    Unit res = units[0];
    for (size_t i = 1; i < units.size(); ++i)
        res = res * units[i];
    return res;
}

Unit DeriveUnitDiv(const std::vector<Unit>& units, OpCategory /*category*/) {
    Unit res = units[0];
    for (size_t i = 1; i < units.size(); ++i)
        res = res / units[i];
    return res;
}

Unit DeriveUnitDimless(const std::vector<Unit>& /*units*/, OpCategory /*category*/) {
    return Unit();
}

Unit DeriveUnitFirst(const std::vector<Unit>& units, OpCategory /*category*/) {
    return units[0];
}

// =========================================================================
//  Predefined OpTraits (execute = NULL for now)
// =========================================================================

// ---- binary arithmetic -----------------------------------------------------

const OpTraits kOpAdd = {
    OpCategory::kAdd, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitAdd, NULL
};

const OpTraits kOpSub = {
    OpCategory::kSub, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitAdd, NULL
};

const OpTraits kOpMul = {
    OpCategory::kMul, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitMul, NULL
};

const OpTraits kOpDiv = {
    OpCategory::kDiv, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeDiv, DeriveUnitDiv, NULL
};

const OpTraits kOpMod = {
    OpCategory::kMod, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, NULL
};

// ---- binary comparison -----------------------------------------------------

const OpTraits kOpEq = {
    OpCategory::kEq, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpNeq = {
    OpCategory::kNeq, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpLt = {
    OpCategory::kLt, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpGt = {
    OpCategory::kGt, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpLe = {
    OpCategory::kLe, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpGe = {
    OpCategory::kGe, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

// ---- binary logical --------------------------------------------------------

const OpTraits kOpAnd = {
    OpCategory::kAnd, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

const OpTraits kOpOr = {
    OpCategory::kOr, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitDimless, NULL
};

// ---- binary bitwise --------------------------------------------------------

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

// ---- binary shift ----------------------------------------------------------

const OpTraits kOpShl = {
    OpCategory::kShl, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitFirst, NULL
};

const OpTraits kOpShr = {
    OpCategory::kShr, 2, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypeForceInt, DeriveUnitFirst, NULL
};

// ---- unary -----------------------------------------------------------------

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

// ---- ternary ---------------------------------------------------------------

const OpTraits kOpConditional = {
    OpCategory::kConditional, 3, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, NULL
};

// ---- variadic --------------------------------------------------------------

const OpTraits kOpCombine = {
    OpCategory::kCombine, Arity::kVariadic, DeriveShapeBroadcast, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, NULL
};

const OpTraits kOpConcat = {
    OpCategory::kConcat, Arity::kVariadic, DeriveShapeConcat, DeriveRowsBroadcast,
    DeriveDtypePromote, DeriveUnitFirst, NULL
};

}  // namespace xdataset
