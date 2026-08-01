#pragma once

#include <boost/variant.hpp>

#include <memory>
#include <string>
#include "data_array.h"
#include "measurement.h"
#include "xdataset_predefine.h"
#include "unit.h"

namespace xdataset {

// =========================================================================
//  Value — unified value type for Measurement and DataArray
// =========================================================================
//
//  Value is a two-way variant:
//    Measurement                 — scalar / vector / matrix + unit (by value)
//    shared_ptr<DataArray>       — named variable with coordinate axes
//
//  Measurement is stored by value (~64 bytes on the stack); DataArray is
//  stored via shared_ptr to avoid deep copies when the same array is
//  returned through multiple evaluation paths.

class XDATASET_API Value
{
public:
    // ---- construction --------------------------------------------------

    /// Default: Measurement Integer 0
    Value();

    /// Implicit from Measurement.
    Value(Measurement m);  // NOLINT(runtime/explicit)

    /// Implicit from DataArray (wraps in shared_ptr).
    Value(const DataArray& da);  // NOLINT(runtime/explicit)

    /// Implicit from DataArray shared_ptr.
    Value(std::shared_ptr<DataArray> da);  // NOLINT(runtime/explicit)

    // Copy / move: compiler-generated is fine (variant + shared_ptr are
    // both deep-copyable / movable).
    Value(const Value&) = default;
    Value& operator=(const Value&) = default;
    Value(Value&&) = default;
    Value& operator=(Value&&) = default;

    // ---- type queries --------------------------------------------------

    /// True when this Value holds a Measurement.
    bool is_measurement() const;

    /// True when this Value holds a DataArray.
    bool is_data_array() const;

    // ---- accessors (throw boost::bad_get on type mismatch) -------------

    Measurement& as_measurement();
    const Measurement& as_measurement() const;

    DataArray& as_data_array();
    const DataArray& as_data_array() const;

    // ---- unified metadata ----------------------------------------------

    DataKind  data_kind() const;
    DataType  data_type() const;
    DataShape data_shape() const;
    const Unit& unit() const;
    Index     rows() const;         // Measurement = 1, DataArray = data().size()
    Index     element_count() const;

    // ---- convenience queries -------------------------------------------

    bool is_scalar() const { return data_kind() == DataKind::kScalar; }
    bool is_vector() const { return data_kind() == DataKind::kVector; }
    bool is_matrix() const { return data_kind() == DataKind::kMatrix; }

    // ---- canonicalization ----------------------------------------------

    /// Return a canonicalized copy (multiplier absorbed, unit = base SI).
    /// Measurement: delegates to Measurement::canonicalized().
    /// DataArray: canonicalizes kSelf DataSeries, preserves indep dims.
    Value canonicalized() const;

    /// True when already canonical (no-op for canonicalized()).
    bool is_canonicalized() const;

    // ---- formatting ----------------------------------------------------

    /// Human-readable string.
    /// When `name` is empty: Measurement renders inline (e.g. "3.14 GHz"),
    /// DataArray renders as DataFrame with a default header.
    /// When `name` is given: Measurement is wrapped in a named DataFrame;
    /// DataArray uses the name as its header.  `max_rows` caps output rows
    /// (0 = no limit).
    std::string Format(const std::string& name = "data", int max_rows = 32) const;

    // ---- convenience factories -----------------------------------------

    static Value Real(double v);
    static Value Integer(int v);
    static Value BooleanValue(bool b);
    static Value String(const std::string& s);

private:
    typedef boost::variant<
        Measurement,
        std::shared_ptr<DataArray>
    > Storage;
    Storage storage_;
};

// =========================================================================
//  Value operators (delegate to OperationXxx)
// =========================================================================

XDATASET_API Value operator+(const Value& lhs, const Value& rhs);
XDATASET_API Value operator-(const Value& lhs, const Value& rhs);
XDATASET_API Value operator*(const Value& lhs, const Value& rhs);
XDATASET_API Value operator/(const Value& lhs, const Value& rhs);
XDATASET_API Value operator%(const Value& lhs, const Value& rhs);

XDATASET_API Value operator==(const Value& lhs, const Value& rhs);
XDATASET_API Value operator!=(const Value& lhs, const Value& rhs);
XDATASET_API Value operator<(const Value& lhs, const Value& rhs);
XDATASET_API Value operator>(const Value& lhs, const Value& rhs);
XDATASET_API Value operator<=(const Value& lhs, const Value& rhs);
XDATASET_API Value operator>=(const Value& lhs, const Value& rhs);

XDATASET_API Value operator&&(const Value& lhs, const Value& rhs);
XDATASET_API Value operator||(const Value& lhs, const Value& rhs);

XDATASET_API Value operator&(const Value& lhs, const Value& rhs);
XDATASET_API Value operator|(const Value& lhs, const Value& rhs);
XDATASET_API Value operator^(const Value& lhs, const Value& rhs);
XDATASET_API Value operator<<(const Value& lhs, const Value& rhs);
XDATASET_API Value operator>>(const Value& lhs, const Value& rhs);

XDATASET_API Value operator-(const Value& v);
XDATASET_API Value operator!(const Value& v);
XDATASET_API Value operator~(const Value& v);

XDATASET_API Value pow(const Value& base, const Value& exponent);

}  // namespace xdataset
