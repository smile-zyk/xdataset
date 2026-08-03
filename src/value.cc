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

// ---- type queries ----------------------------------------------------------

bool Value::is_measurement() const {
    return storage_.which() == 0;
}

bool Value::is_data_array() const {
    return storage_.which() == 1;
}

// ---- accessors -------------------------------------------------------------

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

// ---- unified metadata ------------------------------------------------------

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

// ---- canonicalization ------------------------------------------------------

Value Value::canonicalized() const
{
    if (is_measurement()) {
        const Measurement& m = as_measurement();
        if (m.is_canonicalized()) return *this;
        return Value(m.canonicalized());
    }
    if (is_data_array()) {
        const DataArray& da = as_data_array();
        if (da.data().is_canonicalized()) return *this;

        auto canonical_datas = da.datas();
        canonical_datas[DataArray::kSelf] = da.data().canonicalized();

        DataArrayCreateInfo info;
        info.datas                = std::move(canonical_datas);
        info.multi_dimension_spec = da.multi_dimension_spec();
        info.kind                 = da.data_kind();

        return Value(std::make_shared<DataArray>(std::move(info)));
    }
    return *this;
}

bool Value::is_canonicalized() const
{
    if (is_measurement()) return as_measurement().is_canonicalized();
    if (is_data_array()) return as_data_array().data().is_canonicalized();
    return true;
}

// ---- formatting ------------------------------------------------------------

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

// ---- convenience factories -------------------------------------------------

Value Value::Real(double v, const Unit& u) {
    return Value(Measurement(v, u));
}

Value Value::Integer(int v, const Unit& u) {
    return Value(Measurement(v, u));
}

Value Value::Boolean(bool b) {
    return Value(Measurement::Boolean(b));
}

Value Value::String(const std::string& s) {
    return Value(Measurement::String(s));
}

Value Value::Complex(std::complex<double> v, const Unit& u) {
    return Value(Measurement(v, u));
}

Value Value::Vector(const VecXd& v, const Unit& u) {
    return Value(Measurement(v, u));
}

Value Value::Vector(const VecXi& v, const Unit& u) {
    return Value(Measurement(v, u));
}

Value Value::Vector(const VecXcd& v, const Unit& u) {
    return Value(Measurement(v, u));
}

Value Value::Vector(const VecXs& v) {
    return Value(Measurement::Vector(v));
}

Value Value::Matrix(const MatXd& m, const Unit& u) {
    return Value(Measurement(m, u));
}

Value Value::Matrix(const MatXi& m, const Unit& u) {
    return Value(Measurement(m, u));
}

Value Value::Matrix(const MatXcd& m, const Unit& u) {
    return Value(Measurement(m, u));
}

Value Value::Matrix(const MatXs& m) {
    return Value(Measurement::Matrix(m));
}

Value Value::ArrayReal(const std::vector<double>& v, const Unit& u) {
    return Value(DataArray::CreateIndependent(
        DataSeries::CreateScalarFromVector<double>(v, u)));
}

Value Value::ArrayInteger(const std::vector<int>& v, const Unit& u) {
    return Value(DataArray::CreateIndependent(
        DataSeries::CreateScalarFromVector<int>(v, u)));
}

Value Value::ArrayComplex(const std::vector<std::complex<double>>& v,
                          const Unit& u) {
    return Value(DataArray::CreateIndependent(
        DataSeries::CreateScalarFromVector<std::complex<double>>(v, u)));
}

Value Value::ArrayString(const std::vector<std::string>& v) {
    return Value(DataArray::CreateIndependent(
        DataSeries::CreateScalarFromVector(v)));
}

Value Value::ArrayVector(const std::vector<VecXd>& rows, const Unit& u) {
    DataSeries s(DataKind::kVector, DataType::kReal,
                 {rows.empty() ? Index(0) : rows[0].size()});
    s.set_unit(u);
    for (const auto& row : rows) s.append(Measurement::Vector(row));
    return Value(DataArray::CreateIndependent(std::move(s)));
}

Value Value::ArrayVector(const std::vector<VecXi>& rows, const Unit& u) {
    DataSeries s(DataKind::kVector, DataType::kInteger,
                 {rows.empty() ? Index(0) : rows[0].size()});
    s.set_unit(u);
    for (const auto& row : rows) s.append(Measurement::Vector(row));
    return Value(DataArray::CreateIndependent(std::move(s)));
}

Value Value::ArrayVector(const std::vector<VecXcd>& rows, const Unit& u) {
    DataSeries s(DataKind::kVector, DataType::kComplex,
                 {rows.empty() ? Index(0) : rows[0].size()});
    s.set_unit(u);
    for (const auto& row : rows) s.append(Measurement::Vector(row));
    return Value(DataArray::CreateIndependent(std::move(s)));
}

Value Value::ArrayVector(const std::vector<VecXs>& rows) {
    DataSeries s(DataKind::kVector, DataType::kString,
                 {rows.empty() ? Index(0) : rows[0].dimension(0)});
    for (const auto& row : rows) s.append(Measurement::Vector(row));
    return Value(DataArray::CreateIndependent(std::move(s)));
}

Value Value::ArrayMatrix(const std::vector<MatXd>& rows, const Unit& u) {
    DataSeries s(DataKind::kMatrix, DataType::kReal,
                 {rows.empty() ? Index(0) : rows[0].rows(),
                  rows.empty() ? Index(0) : rows[0].cols()});
    s.set_unit(u);
    for (const auto& row : rows) s.append(Measurement::Matrix(row));
    return Value(DataArray::CreateIndependent(std::move(s)));
}

Value Value::ArrayMatrix(const std::vector<MatXi>& rows, const Unit& u) {
    DataSeries s(DataKind::kMatrix, DataType::kInteger,
                 {rows.empty() ? Index(0) : rows[0].rows(),
                  rows.empty() ? Index(0) : rows[0].cols()});
    s.set_unit(u);
    for (const auto& row : rows) s.append(Measurement::Matrix(row));
    return Value(DataArray::CreateIndependent(std::move(s)));
}

Value Value::ArrayMatrix(const std::vector<MatXcd>& rows, const Unit& u) {
    DataSeries s(DataKind::kMatrix, DataType::kComplex,
                 {rows.empty() ? Index(0) : rows[0].rows(),
                  rows.empty() ? Index(0) : rows[0].cols()});
    s.set_unit(u);
    for (const auto& row : rows) s.append(Measurement::Matrix(row));
    return Value(DataArray::CreateIndependent(std::move(s)));
}

Value Value::ArrayMatrix(const std::vector<MatXs>& rows) {
    DataSeries s(DataKind::kMatrix, DataType::kString,
                 {rows.empty() ? Index(0) : rows[0].dimension(0),
                  rows.empty() ? Index(0) : rows[0].dimension(1)});
    for (const auto& row : rows) s.append(Measurement::Matrix(row));
    return Value(DataArray::CreateIndependent(std::move(s)));
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
