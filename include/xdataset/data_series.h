#ifndef XDATASET_DATA_SERIES_H_
#define XDATASET_DATA_SERIES_H_

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "detail/storage.h"
#include "measurement.h"
#include "units_util.h"
#include "xdataset_predefine.h"

namespace xdataset {

class DataSeries;

class DataSeries {
public:
    class RowView;
    class ConstRowView;

    class iterator {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef RowView value_type;
        typedef std::ptrdiff_t difference_type;
        typedef void pointer;
        typedef RowView reference;

        iterator() : owner_(nullptr), idx_(0) {}
        iterator(DataSeries* owner, Index idx) : owner_(owner), idx_(idx) {}

        reference operator*() const;

        iterator& operator++() {
            ++idx_;
            return *this;
        }

        iterator operator++(int) {
            iterator tmp(*this);
            ++(*this);
            return tmp;
        }

        bool operator==(const iterator& rhs) const {
            return owner_ == rhs.owner_ && idx_ == rhs.idx_;
        }

        bool operator!=(const iterator& rhs) const { return !(*this == rhs); }

    private:
        DataSeries* owner_;
        Index idx_;
    };

    class const_iterator {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef ConstRowView value_type;
        typedef std::ptrdiff_t difference_type;
        typedef void pointer;
        typedef ConstRowView reference;

        const_iterator() : owner_(nullptr), idx_(0) {}
        const_iterator(const DataSeries* owner, Index idx) : owner_(owner), idx_(idx) {}

        reference operator*() const;

        const_iterator& operator++() {
            ++idx_;
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator tmp(*this);
            ++(*this);
            return tmp;
        }

        bool operator==(const const_iterator& rhs) const {
            return owner_ == rhs.owner_ && idx_ == rhs.idx_;
        }

        bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }

    private:
        const DataSeries* owner_;
        Index idx_;
    };

    DataSeries()
        : kind_(DataKind::kScalar),
          dtype_(DTypeTag::kReal),
          shape_(),
          storage_(make_storage(DataKind::kScalar, DTypeTag::kReal, std::vector<Index>())),
          unit_() {}

    DataSeries(DataKind kind, DTypeTag dtype, const std::vector<Index>& shape)
        : kind_(kind), dtype_(dtype), shape_(shape), storage_(make_storage(kind, dtype, shape)), unit_() {}

    DataSeries(const DataSeries& other)
        : kind_(other.kind_), dtype_(other.dtype_), shape_(other.shape_),
          storage_(other.storage_->clone()), unit_(other.unit_) {}

    DataSeries& operator=(const DataSeries& other) {
        if (this != &other) {
            kind_ = other.kind_;
            dtype_ = other.dtype_;
            shape_ = other.shape_;
            storage_ = other.storage_->clone();
            unit_ = other.unit_;
        }
        return *this;
    }

    DataSeries(DataSeries&& other) noexcept
        : kind_(other.kind_), dtype_(other.dtype_),
          shape_(std::move(other.shape_)),
          storage_(std::move(other.storage_)),
          unit_(std::move(other.unit_)) {}

    DataSeries& operator=(DataSeries&& other) noexcept {
        kind_ = other.kind_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        storage_ = std::move(other.storage_);
        unit_ = std::move(other.unit_);
        return *this;
    }

    // -----------------------------------------------------------------------
    //  Factory: Create* — pre-allocate rows (n = 0 → no allocation)
    // -----------------------------------------------------------------------

    template <typename T>
    static DataSeries CreateScalar(std::size_t rows = 0, const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        DataSeries s(DataKind::kScalar, DTypeOf<T>::tag, {});
        s.resize(rows);
        if (rows > 0) s.fill(fill_val);
        return s;
    }

    template <typename T>
    static DataSeries CreateVector(Index width, std::size_t rows = 0,
                                   const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        DataSeries s(DataKind::kVector, DTypeOf<T>::tag, {width});
        s.resize(rows);
        if (rows > 0) s.fill(fill_val);
        return s;
    }

    template <typename T>
    static DataSeries CreateMatrix(Index cell_rows, Index cell_cols,
                                   std::size_t n = 0, const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        DataSeries s(DataKind::kMatrix, DTypeOf<T>::tag, {cell_rows, cell_cols});
        s.resize(n);
        if (n > 0) s.fill(fill_val);
        return s;
    }

    // -----------------------------------------------------------------------
    //  Factory: Create*FromMemory — raw-pointer bulk copy (not for string)
    // -----------------------------------------------------------------------

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, DataSeries>::type
    CreateScalarFromMemory(const T* values, std::size_t len) {
        static_assert(IsSupported<T>::value, "unsupported type");
        if (len > 0 && values == nullptr)
            throw std::invalid_argument("values pointer must not be null when len > 0");
        DataSeries s(DataKind::kScalar, DTypeOf<T>::tag, {});
        s.resize(len);
        std::copy(values, values + len, s.mutable_contiguous_data<T>());
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, DataSeries>::type
    CreateVectorFromMemory(Index width, const T* values, std::size_t len) {
        if (width < 0)
            throw std::invalid_argument("vector width must be non-negative");
        if (len > 0 && values == nullptr)
            throw std::invalid_argument("values pointer must not be null when len > 0");
        const std::size_t w = static_cast<std::size_t>(width);
        if (w == 0) {
            if (len != 0)
                throw std::invalid_argument("vector width 0 requires len = 0");
            return DataSeries(DataKind::kVector, DTypeOf<T>::tag, {width});
        }
        if (len % w != 0)
            throw std::invalid_argument("vector flat data length must be a multiple of width");
        DataSeries s(DataKind::kVector, DTypeOf<T>::tag, {width});
        s.resize(len / w);
        std::copy(values, values + len, s.mutable_contiguous_data<T>());
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, DataSeries>::type
    CreateMatrixFromMemory(Index cell_rows, Index cell_cols, const T* values, std::size_t len) {
        if (cell_rows < 0 || cell_cols < 0)
            throw std::invalid_argument("matrix shape must be non-negative");
        if (len > 0 && values == nullptr)
            throw std::invalid_argument("values pointer must not be null when len > 0");
        const std::size_t elems = static_cast<std::size_t>(cell_rows) *
                                  static_cast<std::size_t>(cell_cols);
        if (elems == 0) {
            if (len != 0)
                throw std::invalid_argument("zero-sized matrix cells require len = 0");
            return DataSeries(DataKind::kMatrix, DTypeOf<T>::tag, {cell_rows, cell_cols});
        }
        if (len % elems != 0)
            throw std::invalid_argument("matrix flat data length must be a multiple of cell_rows * cell_cols");
        DataSeries s(DataKind::kMatrix, DTypeOf<T>::tag, {cell_rows, cell_cols});
        s.resize(len / elems);
        std::copy(values, values + len, s.mutable_contiguous_data<T>());
        return s;
    }

    // -----------------------------------------------------------------------
    //  Factory: Create*FromVector — flat vector-based
    // -----------------------------------------------------------------------

    template <typename T>
    static DataSeries CreateScalarFromVector(const std::vector<T>& values) {
        return CreateScalarFromMemory<T>(
            values.empty() ? static_cast<const T*>(nullptr) : &values[0],
            values.size());
    }

    static DataSeries CreateScalarFromVector(const std::vector<std::string>& values) {
        DataSeries s(DataKind::kScalar, DTypeTag::kString, {});
        s.resize(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            s.scalar_at<std::string>(static_cast<Index>(i)) = values[i];
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, DataSeries>::type
    CreateVectorFromVector(Index width, const std::vector<T>& values) {
        return CreateVectorFromMemory<T>(width,
            values.empty() ? static_cast<const T*>(nullptr) : &values[0],
            values.size());
    }

    static DataSeries CreateVectorFromVector(Index width, const std::vector<std::string>& values) {
        if (width < 0)
            throw std::invalid_argument("vector width must be non-negative");
        const std::size_t w = static_cast<std::size_t>(width);
        if (w == 0) {
            if (!values.empty())
                throw std::invalid_argument("vector width 0 requires empty data");
            return DataSeries(DataKind::kVector, DTypeTag::kString, {width});
        }
        if (values.size() % w != 0)
            throw std::invalid_argument("vector flat data length must be a multiple of width");
        DataSeries s(DataKind::kVector, DTypeTag::kString, {width});
        s.resize(values.size() / w);
        for (std::size_t i = 0; i < s.size(); ++i)
            for (Index j = 0; j < width; ++j)
                s.vector_at<std::string>(static_cast<Index>(i))(j) = values[i * w + j];
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, DataSeries>::type
    CreateMatrixFromVector(Index cell_rows, Index cell_cols, const std::vector<T>& values) {
        return CreateMatrixFromMemory<T>(cell_rows, cell_cols,
            values.empty() ? static_cast<const T*>(nullptr) : &values[0],
            values.size());
    }

    static DataSeries CreateMatrixFromVector(Index cell_rows, Index cell_cols,
                                             const std::vector<std::string>& values) {
        if (cell_rows < 0 || cell_cols < 0)
            throw std::invalid_argument("matrix shape must be non-negative");
        const std::size_t elems = static_cast<std::size_t>(cell_rows) *
                                  static_cast<std::size_t>(cell_cols);
        if (elems == 0) {
            if (!values.empty())
                throw std::invalid_argument("zero-sized matrix cells require empty data");
            return DataSeries(DataKind::kMatrix, DTypeTag::kString, {cell_rows, cell_cols});
        }
        if (values.size() % elems != 0)
            throw std::invalid_argument("matrix flat data length must be a multiple of cell_rows * cell_cols");
        DataSeries s(DataKind::kMatrix, DTypeTag::kString, {cell_rows, cell_cols});
        s.resize(values.size() / elems);
        for (std::size_t i = 0; i < s.size(); ++i)
            for (Index r = 0; r < cell_rows; ++r)
                for (Index c = 0; c < cell_cols; ++c)
                    s.matrix_at<std::string>(static_cast<Index>(i))(r, c) =
                        values[i * elems + static_cast<std::size_t>(r) *
                         static_cast<std::size_t>(cell_cols) + static_cast<std::size_t>(c)];
        return s;
    }

    // -----------------------------------------------------------------------
    //  Factory: Create*FromNestedVector — nested vector-of-vectors
    // -----------------------------------------------------------------------

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, DataSeries>::type
    CreateVectorFromNestedVector(const std::vector<std::vector<T>>& rows) {
        if (rows.empty())
            return DataSeries(DataKind::kVector, DTypeOf<T>::tag, {0});
        const Index width = static_cast<Index>(rows[0].size());
        DataSeries s(DataKind::kVector, DTypeOf<T>::tag, {width});
        s.resize(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (static_cast<Index>(rows[i].size()) != width)
                throw std::invalid_argument("all vector rows must have the same width");
            for (Index j = 0; j < width; ++j)
                s.vector_at<T>(static_cast<Index>(i))(j) = rows[i][static_cast<std::size_t>(j)];
        }
        return s;
    }

    static DataSeries CreateVectorFromNestedVector(const std::vector<std::vector<std::string>>& rows) {
        if (rows.empty())
            return DataSeries(DataKind::kVector, DTypeTag::kString, {0});
        const Index width = static_cast<Index>(rows[0].size());
        DataSeries s(DataKind::kVector, DTypeTag::kString, {width});
        s.resize(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (static_cast<Index>(rows[i].size()) != width)
                throw std::invalid_argument("all vector rows must have the same width");
            for (Index j = 0; j < width; ++j)
                s.vector_at<std::string>(static_cast<Index>(i))(j) = rows[i][static_cast<std::size_t>(j)];
        }
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, DataSeries>::type
    CreateMatrixFromNestedVector(const std::vector<std::vector<std::vector<T>>>& rows) {
        if (rows.empty())
            return DataSeries(DataKind::kMatrix, DTypeOf<T>::tag, {0, 0});
        const Index cell_rows = static_cast<Index>(rows[0].size());
        const Index cell_cols = cell_rows > 0 ? static_cast<Index>(rows[0][0].size()) : 0;
        DataSeries s(DataKind::kMatrix, DTypeOf<T>::tag, {cell_rows, cell_cols});
        s.resize(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (static_cast<Index>(rows[i].size()) != cell_rows)
                throw std::invalid_argument("all matrix rows must have the same shape");
            for (Index r = 0; r < cell_rows; ++r) {
                if (static_cast<Index>(rows[i][static_cast<std::size_t>(r)].size()) != cell_cols)
                    throw std::invalid_argument("all matrix rows must have the same shape");
                for (Index c = 0; c < cell_cols; ++c)
                    s.matrix_at<T>(static_cast<Index>(i))(r, c) =
                        rows[i][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            }
        }
        return s;
    }

    static DataSeries CreateMatrixFromNestedVector(
        const std::vector<std::vector<std::vector<std::string>>>& rows) {
        if (rows.empty())
            return DataSeries(DataKind::kMatrix, DTypeTag::kString, {0, 0});
        const Index cell_rows = static_cast<Index>(rows[0].size());
        const Index cell_cols = cell_rows > 0 ? static_cast<Index>(rows[0][0].size()) : 0;
        DataSeries s(DataKind::kMatrix, DTypeTag::kString, {cell_rows, cell_cols});
        s.resize(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (static_cast<Index>(rows[i].size()) != cell_rows)
                throw std::invalid_argument("all matrix rows must have the same shape");
            for (Index r = 0; r < cell_rows; ++r) {
                if (static_cast<Index>(rows[i][static_cast<std::size_t>(r)].size()) != cell_cols)
                    throw std::invalid_argument("all matrix rows must have the same shape");
                for (Index c = 0; c < cell_cols; ++c)
                    s.matrix_at<std::string>(static_cast<Index>(i))(r, c) =
                        rows[i][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            }
        }
        return s;
    }

    std::size_t size() const { return storage_->size(); }
    bool empty() const { return size() == 0; }

    DataKind data_kind() const { return kind_; }
    DTypeTag dtype() const { return dtype_; }
    std::vector<Index> data_shape() const { return shape_; }
    const Unit& unit() const { return unit_; }

    //---- unit assignment & canonicalisation ----------------------------

    // Assign a unit to this series.  No value conversion is performed ��
    // the existing numbers are simply tagged with the given unit.
    void set_unit(const Unit& u) { unit_ = u; }

    void set_unit(const std::string& s) {
        unit_ = parse_unit(s);
    }

    // Convert values in-place to canonical SI (multiplier absorbed,
    // unit = base_units).  Affine units (degC/degF) are handled via
    // units::convert.  Fast path when multiplier == 1 and non-affine.
    void canonicalize() {
        const double mult = multiplier_of(unit_);
        const Unit target = xdataset::canonicalize(unit_);
        const bool affine = is_affine(unit_);

        if (!affine && mult == 1.0) {
            unit_ = target;
            return;
        }

        for (std::size_t i = 0; i < size(); ++i) {
            Index idx = static_cast<Index>(i);
            if (kind_ == DataKind::kScalar) {
                double& v = scalar_at<double>(idx);
                v = affine ? units::convert(v, unit_, target) : v * mult;
            } else if (kind_ == DataKind::kVector) {
                typename NumericVectorTypes<double>::MapType v = vector_at<double>(idx);
                for (Index j = 0; j < v.size(); ++j) {
                    v(j) = affine ? units::convert(v(j), unit_, target) : v(j) * mult;
                }
            } else {
                typename NumericMatrixTypes<double>::MapType m = matrix_at<double>(idx);
                for (Index r = 0; r < m.rows(); ++r) {
                    for (Index c = 0; c < m.cols(); ++c) {
                        m(r, c) = affine ? units::convert(m(r, c), unit_, target) : m(r, c) * mult;
                    }
                }
            }
        }
        unit_ = target;
    }

    // Return a canonicalised copy without modifying *this.
    DataSeries canonicalized() const {
        DataSeries result(*this);
        result.canonicalize();
        return result;
    }

    //---------------------------------------------------------------------

    Index values_per_row() const {
        if (kind_ == DataKind::kScalar) return 1;
        if (kind_ == DataKind::kVector) return shape_[0];
        return shape_[0] * shape_[1];
    }

    std::string dtype_name() const {
        switch (dtype_) {
            case DTypeTag::kReal: return "Real";
            case DTypeTag::kInteger: return "Integer";
            case DTypeTag::kComplex: return "Complex";
            case DTypeTag::kString: return "String";
        }
        return "Unknown";
    }

    void resize(std::size_t n) { storage_->resize(n); }
    void clear() { storage_->resize(0); }

    DataSeries head(std::size_t n) const {
        std::size_t sz = size();
        return iloc(0, n < sz ? n : sz);
    }

    DataSeries tail(std::size_t n) const {
        std::size_t sz = size();
        return iloc(n < sz ? sz - n : 0, sz);
    }

    DataSeries iloc(std::size_t start, std::size_t end) const {
        if (start > end || end > size()) throw std::out_of_range("iloc out of range");
        DataSeries out(kind_, dtype_, shape_);
        out.unit_ = unit_;
        for (std::size_t i = start; i < end; ++i) out.append_from(*this, static_cast<Index>(i));
        return out;
    }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, static_cast<Index>(size())); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, static_cast<Index>(size())); }

    RowView row(Index i);
    ConstRowView row(Index i) const;

    template <typename T>
    T& scalar_at(Index i) {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return scalar_storage<T>()->value(i);
    }

    template <typename T>
    const T& scalar_at(Index i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return scalar_storage<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::MapType>::type
    vector_at(Index i) {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return vector_storage_numeric<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::ConstMapType>::type
    vector_at(Index i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return vector_storage_numeric<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, Eigen::Tensor<std::string, 1>&>::type
    vector_at(Index i) {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return vector_storage_string()->value(i);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 1>&>::type
    vector_at(Index i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return vector_storage_string()->value(i);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::MapType>::type
    matrix_at(Index i) {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return matrix_storage_numeric<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::ConstMapType>::type
    matrix_at(Index i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return matrix_storage_numeric<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, Eigen::Tensor<std::string, 2>&>::type
    matrix_at(Index i) {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return matrix_storage_string()->value(i);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 2>&>::type
    matrix_at(Index i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("row index out of range");
        return matrix_storage_string()->value(i);
    }

    // ---- append helpers (internal) -----------------------------------

    template <typename T>
    void append_scalar(const T& val) {
        scalar_storage<T>()->append(val);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, void>::type
    append_vector(const typename NumericVectorTypes<T>::OwnedType& v) {
        if (v.size() != values_per_row()) throw std::bad_cast();
        vector_storage_numeric<T>()->append(v);
    }

    void append_vector(const Eigen::Tensor<std::string, 1>& v) {
        if (v.dimension(0) != values_per_row()) throw std::bad_cast();
        vector_storage_string()->append(v);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, void>::type
    append_matrix(const typename NumericMatrixTypes<T>::OwnedType& m) {
        if (m.rows() != shape_[0] || m.cols() != shape_[1]) throw std::bad_cast();
        matrix_storage_numeric<T>()->append(m);
    }

    void append_matrix(const Eigen::Tensor<std::string, 2>& m) {
        if (m.dimension(0) != shape_[0] || m.dimension(1) != shape_[1]) throw std::bad_cast();
        matrix_storage_string()->append(m);
    }

    void append_from(const DataSeries& src, Index row) {
        if (src.kind_ != kind_ || src.dtype_ != dtype_ || src.shape_ != shape_) throw std::bad_cast();
        if (row < 0 || static_cast<std::size_t>(row) >= src.size()) throw std::out_of_range("row index out of range");

        if (!same_dimension(src.unit_, unit_))
            throw std::invalid_argument(
                "unit mismatch: series has dimension [" + unit_string(unit_) +
                "], source has [" + unit_string(src.unit_) + "]");

        if (kind_ == DataKind::kScalar) {
            if (dtype_ == DTypeTag::kReal) append_scalar<double>(src.scalar_at<double>(row));
            else if (dtype_ == DTypeTag::kInteger) append_scalar<int>(src.scalar_at<int>(row));
            else if (dtype_ == DTypeTag::kComplex) append_scalar<std::complex<double> >(src.scalar_at<std::complex<double> >(row));
            else append_scalar<std::string>(src.scalar_at<std::string>(row));
            return;
        }

        if (kind_ == DataKind::kVector) {
            if (dtype_ == DTypeTag::kReal) append_vector<double>(src.vector_at<double>(row));
            else if (dtype_ == DTypeTag::kInteger) append_vector<int>(src.vector_at<int>(row));
            else if (dtype_ == DTypeTag::kComplex) append_vector<std::complex<double> >(src.vector_at<std::complex<double> >(row));
            else append_vector(src.vector_at<std::string>(row));
            return;
        }

        if (dtype_ == DTypeTag::kReal) append_matrix<double>(src.matrix_at<double>(row));
        else if (dtype_ == DTypeTag::kInteger) append_matrix<int>(src.matrix_at<int>(row));
        else if (dtype_ == DTypeTag::kComplex) append_matrix<std::complex<double> >(src.matrix_at<std::complex<double> >(row));
        else append_matrix(src.matrix_at<std::string>(row));
    }

    void assign_from(const DataSeries& src, Index src_row, Index dst_row) {
        if (src.kind_ != kind_ || src.dtype_ != dtype_ || src.shape_ != shape_) throw std::bad_cast();
        if (src_row < 0 || static_cast<std::size_t>(src_row) >= src.size() ||
            dst_row < 0 || static_cast<std::size_t>(dst_row) >= size()) throw std::out_of_range("row index out of range");

        if (kind_ == DataKind::kScalar) {
            if (dtype_ == DTypeTag::kReal) scalar_at<double>(dst_row) = src.scalar_at<double>(src_row);
            else if (dtype_ == DTypeTag::kInteger) scalar_at<int>(dst_row) = src.scalar_at<int>(src_row);
            else if (dtype_ == DTypeTag::kComplex) {
                scalar_at<std::complex<double> >(dst_row) = src.scalar_at<std::complex<double> >(src_row);
            } else {
                scalar_at<std::string>(dst_row) = src.scalar_at<std::string>(src_row);
            }
            return;
        }

        if (kind_ == DataKind::kVector) {
            if (dtype_ == DTypeTag::kReal) vector_at<double>(dst_row) = src.vector_at<double>(src_row);
            else if (dtype_ == DTypeTag::kInteger) vector_at<int>(dst_row) = src.vector_at<int>(src_row);
            else if (dtype_ == DTypeTag::kComplex) {
                vector_at<std::complex<double> >(dst_row) = src.vector_at<std::complex<double> >(src_row);
            } else {
                vector_at<std::string>(dst_row) = src.vector_at<std::string>(src_row);
            }
            return;
        }

        if (dtype_ == DTypeTag::kReal) matrix_at<double>(dst_row) = src.matrix_at<double>(src_row);
        else if (dtype_ == DTypeTag::kInteger) matrix_at<int>(dst_row) = src.matrix_at<int>(src_row);
        else if (dtype_ == DTypeTag::kComplex) {
            matrix_at<std::complex<double> >(dst_row) = src.matrix_at<std::complex<double> >(src_row);
        } else {
            matrix_at<std::string>(dst_row) = src.matrix_at<std::string>(src_row);
        }
    }

    void append(const Measurement& m) {
        if (m.kind() != kind_ || m.dtype() != dtype_ || m.shape() != shape_) throw std::bad_cast();

        // First non-dimensionless measurement sets the series' unit.
        if (same_dimension(unit_, Unit()) && !same_dimension(m.unit(), Unit())) {
            unit_ = m.unit();
        } else if (!same_dimension(m.unit(), unit_)) {
            throw std::invalid_argument(
                "unit mismatch: series has dimension [" + unit_string(unit_) +
                "], measurement has [" + unit_string(m.unit()) + "]");
        }

        if (kind_ == DataKind::kScalar) {
            if (dtype_ == DTypeTag::kReal) append_scalar(boost::get<double>(m.storage()));
            else if (dtype_ == DTypeTag::kInteger) append_scalar(boost::get<int>(m.storage()));
            else if (dtype_ == DTypeTag::kComplex) append_scalar(boost::get<std::complex<double> >(m.storage()));
            else append_scalar(boost::get<std::string>(m.storage()));
            return;
        }

        if (kind_ == DataKind::kVector) {
            if (dtype_ == DTypeTag::kReal) append_vector<double>(boost::get<Eigen::VectorXd>(m.storage()));
            else if (dtype_ == DTypeTag::kInteger) append_vector<int>(boost::get<Eigen::VectorXi>(m.storage()));
            else if (dtype_ == DTypeTag::kComplex) append_vector<std::complex<double> >(boost::get<Eigen::VectorXcd>(m.storage()));
            else append_vector(boost::get<Eigen::Tensor<std::string, 1> >(m.storage()));
            return;
        }

        if (dtype_ == DTypeTag::kReal) append_matrix<double>(NumericMatrixTypes<double>::OwnedType(boost::get<Eigen::MatrixXd>(m.storage())));
        else if (dtype_ == DTypeTag::kInteger) append_matrix<int>(NumericMatrixTypes<int>::OwnedType(boost::get<Eigen::MatrixXi>(m.storage())));
        else if (dtype_ == DTypeTag::kComplex) append_matrix<std::complex<double> >(NumericMatrixTypes<std::complex<double> >::OwnedType(boost::get<Eigen::MatrixXcd>(m.storage())));
        else append_matrix(boost::get<Eigen::Tensor<std::string, 2> >(m.storage()));
    }

    template <typename T>
    void fill(const T& val) {
        if (dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        for (std::size_t i = 0; i < size(); ++i) {
            if (kind_ == DataKind::kScalar) {
                scalar_at<T>(static_cast<Index>(i)) = val;
            } else if (kind_ == DataKind::kVector) {
                fill_vector_row(static_cast<Index>(i), val, std::is_same<T, std::string>());
            } else {
                fill_matrix_row(static_cast<Index>(i), val, std::is_same<T, std::string>());
            }
        }
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, T*>::type mutable_contiguous_data() {
        if (dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        if (kind_ == DataKind::kScalar) return scalar_storage<T>()->data();
        if (kind_ == DataKind::kVector) return vector_storage_numeric<T>()->data();
        return matrix_storage_numeric<T>()->data();
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, const T*>::type contiguous_data() const {
        if (dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        if (kind_ == DataKind::kScalar) return scalar_storage<T>()->data();
        if (kind_ == DataKind::kVector) return vector_storage_numeric<T>()->data();
        return matrix_storage_numeric<T>()->data();
    }

    std::size_t contiguous_elements() const {
        return size() * static_cast<std::size_t>(values_per_row());
    }

    bool is_trivially_copyable() const {
        return dtype_ != DTypeTag::kString;
    }

    std::size_t contiguous_bytes() const {
        if (dtype_ == DTypeTag::kReal) return contiguous_elements() * sizeof(double);
        if (dtype_ == DTypeTag::kInteger) return contiguous_elements() * sizeof(int);
        if (dtype_ == DTypeTag::kComplex) return contiguous_elements() * sizeof(std::complex<double>);
        throw std::runtime_error("string storage is not trivially-copyable");
    }

    Measurement measurement_at(Index i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("index out of range");

        if (kind_ == DataKind::kScalar) {
            if (dtype_ == DTypeTag::kReal) return Measurement(scalar_at<double>(i), unit_);
            else if (dtype_ == DTypeTag::kInteger) return Measurement(scalar_at<int>(i), unit_);
            else if (dtype_ == DTypeTag::kComplex) return Measurement(scalar_at<std::complex<double> >(i), unit_);
            else return Measurement(scalar_at<std::string>(i), unit_);
        }

        if (kind_ == DataKind::kVector) {
            if (dtype_ == DTypeTag::kReal) return Measurement(Eigen::VectorXd(vector_at<double>(i)), unit_);
            else if (dtype_ == DTypeTag::kInteger) return Measurement(Eigen::VectorXi(vector_at<int>(i)), unit_);
            else if (dtype_ == DTypeTag::kComplex) return Measurement(Eigen::VectorXcd(vector_at<std::complex<double> >(i)), unit_);
            else return Measurement(Eigen::Tensor<std::string, 1>(vector_at<std::string>(i)), unit_);
        }

        if (dtype_ == DTypeTag::kReal)
            return Measurement(Eigen::MatrixXd(matrix_at<double>(i)), unit_);
        if (dtype_ == DTypeTag::kInteger)
            return Measurement(Eigen::MatrixXi(matrix_at<int>(i)), unit_);
        if (dtype_ == DTypeTag::kComplex)
            return Measurement(Eigen::MatrixXcd(matrix_at<std::complex<double> >(i)), unit_);
        return Measurement(Eigen::Tensor<std::string, 2>(matrix_at<std::string>(i)), unit_);
    }

    DataSeries at(const std::vector<Index>& selected, bool reduce_to_scalar = false) const {
        if (kind_ == DataKind::kScalar) {
            throw std::logic_error("at is invalid for scalar data");
        }

        if (kind_ == DataKind::kVector) {
            if (reduce_to_scalar) {
                if (selected.size() != 1) {
                    throw std::invalid_argument("scalar vector at requires exactly one index");
                }

                if (dtype_ == DTypeTag::kReal) {
                    DataSeries out(DataKind::kScalar, DTypeTag::kReal, {});
                    out.resize(size());
                    for (std::size_t row = 0; row < size(); ++row) {
                        out.scalar_at<double>(static_cast<Index>(row)) = vector_at<double>(static_cast<Index>(row))(selected[0]);
                    }
                    return out;
                }
                if (dtype_ == DTypeTag::kInteger) {
                    DataSeries out(DataKind::kScalar, DTypeTag::kInteger, {});
                    out.resize(size());
                    for (std::size_t row = 0; row < size(); ++row) {
                        out.scalar_at<int>(static_cast<Index>(row)) = vector_at<int>(static_cast<Index>(row))(selected[0]);
                    }
                    return out;
                }
                if (dtype_ == DTypeTag::kComplex) {
                    DataSeries out(DataKind::kScalar, DTypeTag::kComplex, {});
                    out.resize(size());
                    for (std::size_t row = 0; row < size(); ++row) {
                        out.scalar_at<std::complex<double> >(static_cast<Index>(row)) =
                            vector_at<std::complex<double> >(static_cast<Index>(row))(selected[0]);
                    }
                    return out;
                }

                DataSeries out(DataKind::kScalar, DTypeTag::kString, {});
                out.resize(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    out.scalar_at<std::string>(static_cast<Index>(row)) = vector_at<std::string>(static_cast<Index>(row))(selected[0]);
                }
                return out;
            }

            if (dtype_ == DTypeTag::kReal) {
                return at_vector_numeric_impl<double>(selected);
            }
            if (dtype_ == DTypeTag::kInteger) {
                return at_vector_numeric_impl<int>(selected);
            }
            if (dtype_ == DTypeTag::kComplex) {
                return at_vector_numeric_impl<std::complex<double> >(selected);
            }
            return at_vector_string_impl(selected);
        }

        throw std::invalid_argument("vector at requires vector data");
    }

    DataSeries at(
        const std::vector<Index>& selected_rows,
        const std::vector<Index>& selected_cols,
        bool row_reduce = false,
        bool col_reduce = false) const {
        if (kind_ == DataKind::kScalar) {
            throw std::logic_error("at is invalid for scalar data");
        }

        if (kind_ == DataKind::kVector) {
            throw std::invalid_argument("matrix at requires matrix data");
        }

        if (row_reduce && col_reduce) {
            if (selected_rows.size() != 1 || selected_cols.size() != 1) {
                throw std::invalid_argument("scalar matrix at requires exactly one row and one column index");
            }

            if (dtype_ == DTypeTag::kReal) {
                DataSeries out(DataKind::kScalar, DTypeTag::kReal, {});
                out.resize(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    out.scalar_at<double>(static_cast<Index>(row)) =
                        matrix_at<double>(static_cast<Index>(row))(selected_rows[0], selected_cols[0]);
                }
                return out;
            }
            if (dtype_ == DTypeTag::kInteger) {
                DataSeries out(DataKind::kScalar, DTypeTag::kInteger, {});
                out.resize(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    out.scalar_at<int>(static_cast<Index>(row)) =
                        matrix_at<int>(static_cast<Index>(row))(selected_rows[0], selected_cols[0]);
                }
                return out;
            }
            if (dtype_ == DTypeTag::kComplex) {
                DataSeries out(DataKind::kScalar, DTypeTag::kComplex, {});
                out.resize(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    out.scalar_at<std::complex<double> >(static_cast<Index>(row)) =
                        matrix_at<std::complex<double> >(static_cast<Index>(row))(selected_rows[0], selected_cols[0]);
                }
                return out;
            }

            DataSeries out(DataKind::kScalar, DTypeTag::kString, {});
            out.resize(size());
            for (std::size_t row = 0; row < size(); ++row) {
                out.scalar_at<std::string>(static_cast<Index>(row)) =
                    matrix_at<std::string>(static_cast<Index>(row))(selected_rows[0], selected_cols[0]);
            }
            return out;
        }

        if (row_reduce || col_reduce) {
            const bool select_columns = row_reduce;
            const std::vector<Index>& remaining = select_columns ? selected_cols : selected_rows;

            if (dtype_ == DTypeTag::kReal) {
                DataSeries out(DataKind::kVector, DTypeTag::kReal, {static_cast<Index>(remaining.size())});
                out.resize(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    auto out_vec = out.vector_at<double>(static_cast<Index>(row));
                    for (std::size_t i = 0; i < remaining.size(); ++i) {
                        const Index r = row_reduce ? selected_rows[0] : remaining[i];
                        const Index c = row_reduce ? remaining[i] : selected_cols[0];
                        out_vec(static_cast<Index>(i)) = matrix_at<double>(static_cast<Index>(row))(r, c);
                    }
                }
                return out;
            }
            if (dtype_ == DTypeTag::kInteger) {
                DataSeries out(DataKind::kVector, DTypeTag::kInteger, {static_cast<Index>(remaining.size())});
                out.resize(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    auto out_vec = out.vector_at<int>(static_cast<Index>(row));
                    for (std::size_t i = 0; i < remaining.size(); ++i) {
                        const Index r = row_reduce ? selected_rows[0] : remaining[i];
                        const Index c = row_reduce ? remaining[i] : selected_cols[0];
                        out_vec(static_cast<Index>(i)) = matrix_at<int>(static_cast<Index>(row))(r, c);
                    }
                }
                return out;
            }
            if (dtype_ == DTypeTag::kComplex) {
                DataSeries out(DataKind::kVector, DTypeTag::kComplex, {static_cast<Index>(remaining.size())});
                out.resize(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    auto out_vec = out.vector_at<std::complex<double> >(static_cast<Index>(row));
                    for (std::size_t i = 0; i < remaining.size(); ++i) {
                        const Index r = row_reduce ? selected_rows[0] : remaining[i];
                        const Index c = row_reduce ? remaining[i] : selected_cols[0];
                        out_vec(static_cast<Index>(i)) = matrix_at<std::complex<double> >(static_cast<Index>(row))(r, c);
                    }
                }
                return out;
            }

            DataSeries out(DataKind::kVector, DTypeTag::kString, {static_cast<Index>(remaining.size())});
            out.resize(size());
            for (std::size_t row = 0; row < size(); ++row) {
                auto& out_vec = out.vector_at<std::string>(static_cast<Index>(row));
                for (std::size_t i = 0; i < remaining.size(); ++i) {
                    const Index r = row_reduce ? selected_rows[0] : remaining[i];
                    const Index c = row_reduce ? remaining[i] : selected_cols[0];
                    out_vec(static_cast<Index>(i)) = matrix_at<std::string>(static_cast<Index>(row))(r, c);
                }
            }
            return out;
        }

        if (dtype_ == DTypeTag::kReal) {
            return at_matrix_numeric_impl<double>(selected_rows, selected_cols);
        }
        if (dtype_ == DTypeTag::kInteger) {
            return at_matrix_numeric_impl<int>(selected_rows, selected_cols);
        }
        if (dtype_ == DTypeTag::kComplex) {
            return at_matrix_numeric_impl<std::complex<double> >(selected_rows, selected_cols);
        }
        return at_matrix_string_impl(selected_rows, selected_cols);
    }

private:
    template <typename T>
    DataSeries at_vector_numeric_impl(const std::vector<Index>& selected) const {
        DataSeries out(DataKind::kVector, DTypeOf<T>::tag, {static_cast<Index>(selected.size())});
        out.resize(size());
        for (std::size_t row = 0; row < size(); ++row) {
            auto out_vec = out.vector_at<T>(static_cast<Index>(row));
            for (std::size_t i = 0; i < selected.size(); ++i) {
                out_vec(static_cast<Index>(i)) = vector_at<T>(static_cast<Index>(row))(selected[i]);
            }
        }
        out.unit_ = unit_;
        return out;
    }

    DataSeries at_vector_string_impl(const std::vector<Index>& selected) const {
        DataSeries out(DataKind::kVector, DTypeTag::kString, {static_cast<Index>(selected.size())});
        out.resize(size());
        for (std::size_t row = 0; row < size(); ++row) {
            auto out_vec = out.vector_at<std::string>(static_cast<Index>(row));
            for (std::size_t i = 0; i < selected.size(); ++i) {
                out_vec(static_cast<Index>(i)) = vector_at<std::string>(static_cast<Index>(row))(selected[i]);
            }
        }
        out.unit_ = unit_;
        return out;
    }

    template <typename T>
    DataSeries at_matrix_numeric_impl(
        const std::vector<Index>& selected_rows,
        const std::vector<Index>& selected_cols) const {

        DataSeries out(DataKind::kMatrix, DTypeOf<T>::tag,
                       {static_cast<Index>(selected_rows.size()),
                        static_cast<Index>(selected_cols.size())});
        out.resize(size());

        for (std::size_t row = 0; row < size(); ++row) {
            auto out_mat = out.matrix_at<T>(static_cast<Index>(row));
            for (std::size_t r = 0; r < selected_rows.size(); ++r) {
                for (std::size_t c = 0; c < selected_cols.size(); ++c) {
                    out_mat(static_cast<Index>(r), static_cast<Index>(c)) =
                        matrix_at<T>(static_cast<Index>(row))(selected_rows[r], selected_cols[c]);
                }
            }
        }
        out.unit_ = unit_;
        return out;
    }

    DataSeries at_matrix_string_impl(
        const std::vector<Index>& selected_rows,
        const std::vector<Index>& selected_cols) const {

        DataSeries out(DataKind::kMatrix, DTypeTag::kString,
                       {static_cast<Index>(selected_rows.size()),
                        static_cast<Index>(selected_cols.size())});
        out.resize(size());

        for (std::size_t row = 0; row < size(); ++row) {
            auto& out_mat = out.matrix_at<std::string>(static_cast<Index>(row));
            for (std::size_t r = 0; r < selected_rows.size(); ++r) {
                for (std::size_t c = 0; c < selected_cols.size(); ++c) {
                    out_mat(static_cast<Index>(r), static_cast<Index>(c)) =
                        matrix_at<std::string>(static_cast<Index>(row))(selected_rows[r], selected_cols[c]);
                }
            }
        }
        out.unit_ = unit_;
        return out;
    }

    static void validate_schema(DataKind kind, const std::vector<Index>& shape) {
        if (kind == DataKind::kScalar) {
            if (!shape.empty()) throw std::invalid_argument("scalar schema must have empty shape");
            return;
        }

        if (kind == DataKind::kVector) {
            if (shape.size() != 1 || shape[0] < 0) {
                throw std::invalid_argument("vector schema must have one non-negative dimension");
            }
            return;
        }

        if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0) {
            throw std::invalid_argument("matrix schema must have two non-negative dimensions");
        }
    }

    static std::unique_ptr<CellStorage> make_storage(DataKind kind, DTypeTag dtype, const std::vector<Index>& shape) {
        validate_schema(kind, shape);
        if (kind == DataKind::kScalar) {
            if (dtype == DTypeTag::kReal) return std::unique_ptr<CellStorage>(new ScalarStorage<double>());
            if (dtype == DTypeTag::kInteger) return std::unique_ptr<CellStorage>(new ScalarStorage<int>());
            if (dtype == DTypeTag::kComplex) return std::unique_ptr<CellStorage>(new ScalarStorage<std::complex<double> >());
            return std::unique_ptr<CellStorage>(new ScalarStorage<std::string>());
        }

        if (kind == DataKind::kVector) {
            if (dtype == DTypeTag::kReal) return std::unique_ptr<CellStorage>(new VectorStorageNumeric<double>(shape[0]));
            if (dtype == DTypeTag::kInteger) return std::unique_ptr<CellStorage>(new VectorStorageNumeric<int>(shape[0]));
            if (dtype == DTypeTag::kComplex) return std::unique_ptr<CellStorage>(new VectorStorageNumeric<std::complex<double> >(shape[0]));
            return std::unique_ptr<CellStorage>(new VectorStorageString(shape[0]));
        }

        if (dtype == DTypeTag::kReal) return std::unique_ptr<CellStorage>(new MatrixStorageNumeric<double>(shape[0], shape[1]));
        if (dtype == DTypeTag::kInteger) return std::unique_ptr<CellStorage>(new MatrixStorageNumeric<int>(shape[0], shape[1]));
        if (dtype == DTypeTag::kComplex) return std::unique_ptr<CellStorage>(new MatrixStorageNumeric<std::complex<double> >(shape[0], shape[1]));
        return std::unique_ptr<CellStorage>(new MatrixStorageString(shape[0], shape[1]));
    }

    template <typename T>
    ScalarStorage<T>* scalar_storage() {
        if (kind_ != DataKind::kScalar || dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        return static_cast<ScalarStorage<T>*>(storage_.get());
    }

    template <typename T>
    const ScalarStorage<T>* scalar_storage() const {
        if (kind_ != DataKind::kScalar || dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        return static_cast<const ScalarStorage<T>*>(storage_.get());
    }

    template <typename T>
    VectorStorageNumeric<T>* vector_storage_numeric() {
        if (kind_ != DataKind::kVector || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) throw std::bad_cast();
        return static_cast<VectorStorageNumeric<T>*>(storage_.get());
    }

    template <typename T>
    const VectorStorageNumeric<T>* vector_storage_numeric() const {
        if (kind_ != DataKind::kVector || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) throw std::bad_cast();
        return static_cast<const VectorStorageNumeric<T>*>(storage_.get());
    }

    VectorStorageString* vector_storage_string() {
        if (kind_ != DataKind::kVector || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<VectorStorageString*>(storage_.get());
    }

    const VectorStorageString* vector_storage_string() const {
        if (kind_ != DataKind::kVector || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<const VectorStorageString*>(storage_.get());
    }

    template <typename T>
    MatrixStorageNumeric<T>* matrix_storage_numeric() {
        if (kind_ != DataKind::kMatrix || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) throw std::bad_cast();
        return static_cast<MatrixStorageNumeric<T>*>(storage_.get());
    }

    template <typename T>
    const MatrixStorageNumeric<T>* matrix_storage_numeric() const {
        if (kind_ != DataKind::kMatrix || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) throw std::bad_cast();
        return static_cast<const MatrixStorageNumeric<T>*>(storage_.get());
    }

    MatrixStorageString* matrix_storage_string() {
        if (kind_ != DataKind::kMatrix || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<MatrixStorageString*>(storage_.get());
    }

    const MatrixStorageString* matrix_storage_string() const {
        if (kind_ != DataKind::kMatrix || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<const MatrixStorageString*>(storage_.get());
    }

    template <typename T>
    void fill_vector_row(Index row, const T& val, std::false_type) {
        vector_at<T>(row).setConstant(val);
    }

    void fill_vector_row(Index row, const std::string& val, std::true_type) {
        Eigen::Tensor<std::string, 1>& t = vector_at<std::string>(row);
        for (Index i = 0; i < t.dimension(0); ++i) t(i) = val;
    }

    template <typename T>
    void fill_matrix_row(Index row, const T& val, std::false_type) {
        matrix_at<T>(row).setConstant(val);
    }

    void fill_matrix_row(Index row, const std::string& val, std::true_type) {
        Eigen::Tensor<std::string, 2>& t = matrix_at<std::string>(row);
        for (Index r = 0; r < t.dimension(0); ++r) {
            for (Index c = 0; c < t.dimension(1); ++c) {
                t(r, c) = val;
            }
        }
    }

    DataKind kind_;
    DTypeTag dtype_;
    std::vector<Index> shape_;
    std::unique_ptr<CellStorage> storage_;
    Unit unit_;
};

class DataSeries::RowView {
public:
    RowView(DataSeries* owner, Index idx) : owner_(owner), idx_(idx) {}

    DataKind kind() const { return checked_owner()->data_kind(); }
    DTypeTag dtype() const { return checked_owner()->dtype(); }
    std::string dtype_name() const { return checked_owner()->dtype_name(); }
    std::vector<Index> data_shape() const { return checked_owner()->data_shape(); }
    Index index() const { return idx_; }

    template <typename T>
    T& scalar() {
        return checked_owner()->scalar_at<T>(idx_);
    }

    template <typename T>
    const T& scalar() const {
        return checked_owner()->scalar_at<T>(idx_);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::MapType>::type
    vector() {
        return checked_owner()->vector_at<T>(idx_);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::ConstMapType>::type
    vector() const {
        return checked_owner()->vector_at<T>(idx_);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, Eigen::Tensor<std::string, 1>&>::type
    vector() {
        return checked_owner()->vector_at<std::string>(idx_);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 1>&>::type
    vector() const {
        return checked_owner()->vector_at<std::string>(idx_);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::MapType>::type
    matrix() {
        return checked_owner()->matrix_at<T>(idx_);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::ConstMapType>::type
    matrix() const {
        return checked_owner()->matrix_at<T>(idx_);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, Eigen::Tensor<std::string, 2>&>::type
    matrix() {
        return checked_owner()->matrix_at<std::string>(idx_);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 2>&>::type
    matrix() const {
        return checked_owner()->matrix_at<std::string>(idx_);
    }

    Measurement to_measurement() const { return checked_owner()->measurement_at(idx_); }

private:
    DataSeries* checked_owner() const {
        if (!owner_ || idx_ < 0 || static_cast<std::size_t>(idx_) >= owner_->size()) throw std::out_of_range("row view is invalid");
        return owner_;
    }

    DataSeries* owner_;
    Index idx_;
};

class DataSeries::ConstRowView {
public:
    ConstRowView(const DataSeries* owner, Index idx) : owner_(owner), idx_(idx) {}

    DataKind kind() const { return checked_owner()->data_kind(); }
    DTypeTag dtype() const { return checked_owner()->dtype(); }
    std::string dtype_name() const { return checked_owner()->dtype_name(); }
    std::vector<Index> data_shape() const { return checked_owner()->data_shape(); }
    Index index() const { return idx_; }

    template <typename T>
    const T& scalar() const {
        return checked_owner()->scalar_at<T>(idx_);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::ConstMapType>::type
    vector() const {
        return checked_owner()->vector_at<T>(idx_);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 1>&>::type
    vector() const {
        return checked_owner()->vector_at<std::string>(idx_);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::ConstMapType>::type
    matrix() const {
        return checked_owner()->matrix_at<T>(idx_);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 2>&>::type
    matrix() const {
        return checked_owner()->matrix_at<std::string>(idx_);
    }

    Measurement to_measurement() const { return checked_owner()->measurement_at(idx_); }

private:
    const DataSeries* checked_owner() const {
        if (!owner_ || idx_ < 0 || static_cast<std::size_t>(idx_) >= owner_->size()) throw std::out_of_range("row view is invalid");
        return owner_;
    }

    const DataSeries* owner_;
    Index idx_;
};

inline DataSeries::RowView DataSeries::row(Index i) {
    if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("index out of range");
    return RowView(this, i);
}

inline DataSeries::ConstRowView DataSeries::row(Index i) const {
    if (i < 0 || static_cast<std::size_t>(i) >= size()) throw std::out_of_range("index out of range");
    return ConstRowView(this, i);
}

inline DataSeries::iterator::reference DataSeries::iterator::operator*() const {
    return RowView(owner_, idx_);
}

inline DataSeries::const_iterator::reference DataSeries::const_iterator::operator*() const {
    return ConstRowView(owner_, idx_);
}

}  // namespace xdataset

#endif  // XDATASET_DATA_SERIES_H_
