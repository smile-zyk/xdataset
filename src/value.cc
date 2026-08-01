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

Value::Value(std::shared_ptr<DataArray> da) : storage_(std::move(da)) {}

bool Value::is_measurement() const {
    return storage_.which() == 0;
}

bool Value::is_data_array() const {
    return storage_.which() == 1;
}

Measurement& Value::as_measurement() {
    return boost::get<Measurement>(storage_);
}

const Measurement& Value::as_measurement() const {
    return boost::get<Measurement>(storage_);
}

DataArray& Value::as_data_array() {
    return *boost::get<std::shared_ptr<DataArray>>(storage_);
}

const DataArray& Value::as_data_array() const {
    return *boost::get<std::shared_ptr<DataArray>>(storage_);
}

DataKind Value::data_kind() const {
    if (is_measurement()) return as_measurement().data_kind();
    return as_data_array().data().data_kind();
}

DataType Value::data_type() const {
    if (is_measurement()) return as_measurement().data_type();
    return as_data_array().data().data_type();
}

DataShape Value::data_shape() const {
    if (is_measurement()) return as_measurement().shape();
    return as_data_array().data().data_shape();
}

const Unit& Value::unit() const {
    if (is_measurement()) return as_measurement().unit();
    return as_data_array().data().unit();
}

Index Value::rows() const {
    if (is_measurement()) return 1;
    return static_cast<Index>(as_data_array().data().size());
}

Index Value::element_count() const {
    if (is_measurement()) return as_measurement().element_count();
    return as_data_array().element_count();
}

std::string Value::Format(const std::string& name, int max_rows) const
{
    if (is_measurement())
    {
        const xdataset::Measurement& m = as_measurement();
        return m.to_dataframe(name).to_string(max_rows);
    }

    // DataArray: render with custom or default variable name
    const xdataset::DataArray& da = as_data_array();
    const std::string& header = name.empty() ? "data" : name;
    return da.GetOrCreateDataFrame(header).to_string(max_rows);
}

Value Value::Real(double v) {
    return Value(Measurement(v));
}

Value Value::Integer(int v) {
    return Value(Measurement(v));
}

Value Value::BooleanValue(bool b) {
    return Value(Measurement::Boolean(b));
}

Value Value::String(const std::string& s) {
    return Value(Measurement::String(s));
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

}  // namespace xdataset
