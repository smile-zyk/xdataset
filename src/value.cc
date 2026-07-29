// =============================================================================
//  xdataset -- Value implementation
// =============================================================================

#include "value.h"

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

std::vector<Index> Value::shape() const {
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

}  // namespace xdataset
