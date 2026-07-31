// =============================================================================
//  xdataset -- Value implementation
// =============================================================================

#include "value.h"
#include "operation.h"

namespace xdataset {

// =========================================================================
//  Value
// =========================================================================

Value::Value() : storage_(Measurement()) {}

Value::Value(Measurement m) : storage_(std::move(m)) {}

Value::Value(const DataArray& da)
    : storage_(std::make_shared<DataArray>(da)) {}

bool Value::is_meas() const {
    return storage_.which() == 0;
}

bool Value::is_array() const {
    return storage_.which() == 1;
}

const Measurement& Value::as_meas() const {
    return boost::get<Measurement>(storage_);
}

const DataArray& Value::as_array() const {
    return *boost::get<std::shared_ptr<DataArray> >(storage_);
}

DataKind Value::data_kind() const {
    if (is_meas()) return as_meas().data_kind();
    return as_array().data().data_kind();
}

DataType Value::data_type() const {
    if (is_meas()) return as_meas().data_type();
    return as_array().data().data_type();
}

DataShape Value::shape() const {
    if (is_meas()) return as_meas().shape();
    return as_array().data().data_shape();
}

const Unit& Value::unit() const {
    if (is_meas()) return as_meas().unit();
    return as_array().data().unit();
}

Index Value::rows() const {
    if (is_meas()) return 1;
    return static_cast<Index>(as_array().data().size());
}

// =========================================================================
//  Value operators (via OperationXxx)
// =========================================================================

Value operator+(const Value& a, const Value& b) { return OperationAdd(a,b); }
Value operator-(const Value& a, const Value& b) { return OperationSub(a,b); }
Value operator*(const Value& a, const Value& b) { return OperationMul(a,b); }
Value operator/(const Value& a, const Value& b) { return OperationDiv(a,b); }
Value operator%(const Value& a, const Value& b) { return OperationMod(a,b); }

Value operator==(const Value& a, const Value& b) { return OperationEq(a,b); }
Value operator!=(const Value& a, const Value& b) { return OperationNeq(a,b); }
Value operator<(const Value& a, const Value& b)  { return OperationLt(a,b); }
Value operator>(const Value& a, const Value& b)  { return OperationGt(a,b); }
Value operator<=(const Value& a, const Value& b) { return OperationLe(a,b); }
Value operator>=(const Value& a, const Value& b) { return OperationGe(a,b); }

Value operator&&(const Value& a, const Value& b) { return OperationAnd(a,b); }
Value operator||(const Value& a, const Value& b) { return OperationOr(a,b); }

Value operator&(const Value& a, const Value& b)  { return OperationBitAnd(a,b); }
Value operator|(const Value& a, const Value& b)  { return OperationBitOr(a,b); }
Value operator^(const Value& a, const Value& b)  { return OperationBitXor(a,b); }
Value operator<<(const Value& a, const Value& b) { return OperationShl(a,b); }
Value operator>>(const Value& a, const Value& b) { return OperationShr(a,b); }

Value operator-(const Value& v) { return OperationNegate(v); }
Value operator!(const Value& v) { return OperationNot(v); }
Value operator~(const Value& v) { return OperationBitNot(v); }

Value pow(const Value& base, const Value& exp) { return OperationPow(base, exp); }

Index Value::element_count() const {
    if (is_meas()) return as_meas().element_count();
    return as_array().element_count();
}

}  // namespace xdataset
