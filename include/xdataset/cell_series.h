#ifndef XDATASET_TENSOR_SERIES_H_
#define XDATASET_TENSOR_SERIES_H_

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <complex>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace xdataset {

using Index = Eigen::Index;

enum class CellKind { kScalar, kVector, kMatrix };
enum class DTypeTag { kReal, kInteger, kComplex, kString };

template <typename T>
struct IsSupported : std::false_type {};

template <>
struct IsSupported<double> : std::true_type {};

template <>
struct IsSupported<int> : std::true_type {};

template <>
struct IsSupported<std::complex<double> > : std::true_type {};

template <>
struct IsSupported<std::string> : std::true_type {};

template <typename T>
struct DTypeOf;

template <>
struct DTypeOf<double> {
    static const DTypeTag tag = DTypeTag::kReal;
};

template <>
struct DTypeOf<int> {
    static const DTypeTag tag = DTypeTag::kInteger;
};

template <>
struct DTypeOf<std::complex<double> > {
    static const DTypeTag tag = DTypeTag::kComplex;
};

template <>
struct DTypeOf<std::string> {
    static const DTypeTag tag = DTypeTag::kString;
};

template <typename T>
struct NumericVectorTypes {
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> OwnedType;
    typedef Eigen::Map<OwnedType> MapType;
    typedef Eigen::Map<const OwnedType> ConstMapType;
};

template <typename T>
struct NumericMatrixTypes {
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> OwnedType;
    typedef Eigen::Map<OwnedType> MapType;
    typedef Eigen::Map<const OwnedType> ConstMapType;
};

class CellSeries;

class CellStorage {
public:
    virtual ~CellStorage() {}
    virtual std::size_t size() const = 0;
    virtual void resize(std::size_t rows) = 0;
    virtual std::unique_ptr<CellStorage> clone() const = 0;
};

template <typename T>
class ScalarStorage : public CellStorage {
public:
    std::size_t size() const override { return values_.size(); }

    void resize(std::size_t rows) override {
        values_.resize(rows, T());
    }

    std::unique_ptr<CellStorage> clone() const override {
        return std::unique_ptr<CellStorage>(new ScalarStorage<T>(*this));
    }

    T& value(std::size_t row) { return values_[row]; }
    const T& value(std::size_t row) const { return values_[row]; }

    void append(const T& v) { values_.push_back(v); }

    T* data() { return values_.empty() ? nullptr : &values_[0]; }
    const T* data() const { return values_.empty() ? nullptr : &values_[0]; }

private:
    std::vector<T> values_;
};

template <typename T>
class VectorStorageNumeric : public CellStorage {
public:
    typedef typename NumericVectorTypes<T>::OwnedType OwnedType;
    typedef typename NumericVectorTypes<T>::MapType MapType;
    typedef typename NumericVectorTypes<T>::ConstMapType ConstMapType;

    explicit VectorStorageNumeric(Index width) : width_(width), rows_(0) {}

    std::size_t size() const override { return rows_; }

    void resize(std::size_t rows) override {
        rows_ = rows;
        values_.resize(static_cast<std::size_t>(width_) * rows_, T());
    }

    std::unique_ptr<CellStorage> clone() const override {
        return std::unique_ptr<CellStorage>(new VectorStorageNumeric<T>(*this));
    }

    MapType value(std::size_t row) {
        return MapType(ptr_for_row(row), width_);
    }

    ConstMapType value(std::size_t row) const {
        return ConstMapType(ptr_for_row(row), width_);
    }

    void set(std::size_t row, const OwnedType& v) {
        if (v.size() != width_) throw std::bad_cast();
        value(row) = v;
    }

    void append(const OwnedType& v) {
        if (v.size() != width_) throw std::bad_cast();
        values_.reserve(values_.size() + static_cast<std::size_t>(width_));
        for (Index i = 0; i < width_; ++i) values_.push_back(v(i));
        ++rows_;
    }

    T* data() { return values_.empty() ? nullptr : &values_[0]; }
    const T* data() const { return values_.empty() ? nullptr : &values_[0]; }

private:
    T* ptr_for_row(std::size_t row) {
        return data() + static_cast<std::size_t>(width_) * row;
    }

    const T* ptr_for_row(std::size_t row) const {
        return data() + static_cast<std::size_t>(width_) * row;
    }

    Index width_;
    std::size_t rows_;
    std::vector<T> values_;
};

class VectorStorageString : public CellStorage {
public:
    explicit VectorStorageString(Index width) : width_(width) {}

    std::size_t size() const override { return rows_.size(); }

    void resize(std::size_t rows) override {
        if (rows <= rows_.size()) {
            rows_.resize(rows);
            return;
        }
        rows_.resize(rows, default_row());
    }

    std::unique_ptr<CellStorage> clone() const override {
        return std::unique_ptr<CellStorage>(new VectorStorageString(*this));
    }

    Eigen::Tensor<std::string, 1>& value(std::size_t row) { return rows_[row]; }
    const Eigen::Tensor<std::string, 1>& value(std::size_t row) const { return rows_[row]; }

    void append(const Eigen::Tensor<std::string, 1>& row) { rows_.push_back(row); }

private:
    Eigen::Tensor<std::string, 1> default_row() const {
        Eigen::Tensor<std::string, 1> out(width_);
        return out;
    }

    Index width_;
    std::vector<Eigen::Tensor<std::string, 1> > rows_;
};

template <typename T>
class MatrixStorageNumeric : public CellStorage {
public:
    typedef typename NumericMatrixTypes<T>::OwnedType OwnedType;
    typedef typename NumericMatrixTypes<T>::MapType MapType;
    typedef typename NumericMatrixTypes<T>::ConstMapType ConstMapType;

    MatrixStorageNumeric(Index rows, Index cols)
        : cell_rows_(rows), cell_cols_(cols), row_count_(0) {}

    std::size_t size() const override { return row_count_; }

    void resize(std::size_t rows) override {
        row_count_ = rows;
        values_.resize(static_cast<std::size_t>(cell_rows_ * cell_cols_) * row_count_, T());
    }

    std::unique_ptr<CellStorage> clone() const override {
        return std::unique_ptr<CellStorage>(new MatrixStorageNumeric<T>(*this));
    }

    MapType value(std::size_t row) {
        return MapType(ptr_for_row(row), cell_rows_, cell_cols_);
    }

    ConstMapType value(std::size_t row) const {
        return ConstMapType(ptr_for_row(row), cell_rows_, cell_cols_);
    }

    void set(std::size_t row, const OwnedType& m) {
        if (m.rows() != cell_rows_ || m.cols() != cell_cols_) throw std::bad_cast();
        value(row) = m;
    }

    void append(const OwnedType& m) {
        if (m.rows() != cell_rows_ || m.cols() != cell_cols_) throw std::bad_cast();
        const std::size_t elems = static_cast<std::size_t>(cell_rows_ * cell_cols_);
        values_.reserve(values_.size() + elems);
        for (Index r = 0; r < cell_rows_; ++r) {
            for (Index c = 0; c < cell_cols_; ++c) {
                values_.push_back(m(r, c));
            }
        }
        ++row_count_;
    }

    T* data() { return values_.empty() ? nullptr : &values_[0]; }
    const T* data() const { return values_.empty() ? nullptr : &values_[0]; }

private:
    T* ptr_for_row(std::size_t row) {
        return data() + static_cast<std::size_t>(cell_rows_ * cell_cols_) * row;
    }

    const T* ptr_for_row(std::size_t row) const {
        return data() + static_cast<std::size_t>(cell_rows_ * cell_cols_) * row;
    }

    Index cell_rows_;
    Index cell_cols_;
    std::size_t row_count_;
    std::vector<T> values_;
};

class MatrixStorageString : public CellStorage {
public:
    MatrixStorageString(Index rows, Index cols) : cell_rows_(rows), cell_cols_(cols) {}

    std::size_t size() const override { return rows_.size(); }

    void resize(std::size_t rows) override {
        if (rows <= rows_.size()) {
            rows_.resize(rows);
            return;
        }
        rows_.resize(rows, default_row());
    }

    std::unique_ptr<CellStorage> clone() const override {
        return std::unique_ptr<CellStorage>(new MatrixStorageString(*this));
    }

    Eigen::Tensor<std::string, 2>& value(std::size_t row) { return rows_[row]; }
    const Eigen::Tensor<std::string, 2>& value(std::size_t row) const { return rows_[row]; }

    void append(const Eigen::Tensor<std::string, 2>& row) { rows_.push_back(row); }

private:
    Eigen::Tensor<std::string, 2> default_row() const {
        Eigen::Tensor<std::string, 2> out(cell_rows_, cell_cols_);
        return out;
    }

    Index cell_rows_;
    Index cell_cols_;
    std::vector<Eigen::Tensor<std::string, 2> > rows_;
};

class Cell {
public:
    Cell()
        : kind_(CellKind::kScalar),
          dtype_(DTypeTag::kReal),
          shape_(),
          storage_(new ScalarStorage<double>()) {
        scalar_storage<double>()->resize(1);
    }

    Cell(const Cell& other)
        : kind_(other.kind_),
          dtype_(other.dtype_),
          shape_(other.shape_),
          storage_(other.storage_ ? other.storage_->clone() : std::unique_ptr<CellStorage>()) {}

    Cell& operator=(const Cell& other) {
        if (this != &other) {
            kind_ = other.kind_;
            dtype_ = other.dtype_;
            shape_ = other.shape_;
            storage_ = other.storage_ ? other.storage_->clone() : std::unique_ptr<CellStorage>();
        }
        return *this;
    }

    Cell(Cell&& other) noexcept
        : kind_(other.kind_),
          dtype_(other.dtype_),
          shape_(std::move(other.shape_)),
          storage_(std::move(other.storage_)) {}

    Cell& operator=(Cell&& other) noexcept {
        kind_ = other.kind_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        storage_ = std::move(other.storage_);
        return *this;
    }

    template <typename T>
    static Cell Scalar(const T& value) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Cell c;
        c.kind_ = CellKind::kScalar;
        c.dtype_ = DTypeOf<T>::tag;
        c.shape_.clear();
        c.storage_.reset(new ScalarStorage<T>());
        c.scalar_storage<T>()->resize(1);
        c.scalar_storage<T>()->value(0) = value;
        return c;
    }

    template <typename T>
    static Cell Scalar() {
        return Scalar<T>(T());
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, Cell>::type Vector(
        const typename NumericVectorTypes<T>::OwnedType& row) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Cell c;
        c.kind_ = CellKind::kVector;
        c.dtype_ = DTypeOf<T>::tag;
        c.shape_ = std::vector<Index>(1, static_cast<Index>(row.size()));
        c.storage_.reset(new VectorStorageNumeric<T>(static_cast<Index>(row.size())));
        c.vector_storage_numeric<T>()->resize(1);
        c.vector_storage_numeric<T>()->set(0, row);
        return c;
    }

    static Cell Vector(const Eigen::Tensor<std::string, 1>& row) {
        Cell c;
        c.kind_ = CellKind::kVector;
        c.dtype_ = DTypeTag::kString;
        c.shape_ = std::vector<Index>(1, static_cast<Index>(row.size()));
        c.storage_.reset(new VectorStorageString(static_cast<Index>(row.size())));
        c.vector_storage_string()->resize(1);
        c.vector_storage_string()->value(0) = row;
        return c;
    }

    template <typename T>
    static Cell Vector(Index width) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Cell c;
        c.kind_ = CellKind::kVector;
        c.dtype_ = DTypeOf<T>::tag;
        c.shape_ = std::vector<Index>(1, width);
        if (std::is_same<T, std::string>::value) {
            c.storage_.reset(new VectorStorageString(width));
            c.vector_storage_string()->resize(1);
        } else {
            c.storage_.reset(new VectorStorageNumeric<T>(width));
            c.vector_storage_numeric<T>()->resize(1);
        }
        return c;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, Cell>::type Matrix(
        const typename NumericMatrixTypes<T>::OwnedType& row) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Cell c;
        c.kind_ = CellKind::kMatrix;
        c.dtype_ = DTypeOf<T>::tag;
        c.shape_.push_back(static_cast<Index>(row.rows()));
        c.shape_.push_back(static_cast<Index>(row.cols()));
        c.storage_.reset(new MatrixStorageNumeric<T>(static_cast<Index>(row.rows()), static_cast<Index>(row.cols())));
        c.matrix_storage_numeric<T>()->resize(1);
        c.matrix_storage_numeric<T>()->set(0, row);
        return c;
    }

    static Cell Matrix(const Eigen::Tensor<std::string, 2>& row) {
        Cell c;
        c.kind_ = CellKind::kMatrix;
        c.dtype_ = DTypeTag::kString;
        c.shape_.push_back(row.dimension(0));
        c.shape_.push_back(row.dimension(1));
        c.storage_.reset(new MatrixStorageString(row.dimension(0), row.dimension(1)));
        c.matrix_storage_string()->resize(1);
        c.matrix_storage_string()->value(0) = row;
        return c;
    }

    template <typename T>
    static Cell Matrix(Index rows, Index cols) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Cell c;
        c.kind_ = CellKind::kMatrix;
        c.dtype_ = DTypeOf<T>::tag;
        c.shape_.push_back(rows);
        c.shape_.push_back(cols);
        if (std::is_same<T, std::string>::value) {
            c.storage_.reset(new MatrixStorageString(rows, cols));
            c.matrix_storage_string()->resize(1);
        } else {
            c.storage_.reset(new MatrixStorageNumeric<T>(rows, cols));
            c.matrix_storage_numeric<T>()->resize(1);
        }
        return c;
    }

    bool has_value() const { return storage_.get() != nullptr; }
    CellKind kind() const { return kind_; }
    DTypeTag dtype() const { return dtype_; }
    std::vector<Index> cell_shape() const { return shape_; }

    std::string dtype_name() const {
        switch (dtype_) {
            case DTypeTag::kReal: return "Real";
            case DTypeTag::kInteger: return "Integer";
            case DTypeTag::kComplex: return "Complex";
            case DTypeTag::kString: return "String";
        }
        return "Unknown";
    }

    template <typename T>
    T& scalar() {
        return scalar_storage<T>()->value(0);
    }

    template <typename T>
    const T& scalar() const {
        return scalar_storage<T>()->value(0);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::MapType>::type
    vector() {
        return vector_storage_numeric<T>()->value(0);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::ConstMapType>::type
    vector() const {
        return vector_storage_numeric<T>()->value(0);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, Eigen::Tensor<std::string, 1>&>::type
    vector() {
        return vector_storage_string()->value(0);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 1>&>::type
    vector() const {
        return vector_storage_string()->value(0);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::MapType>::type
    matrix() {
        return matrix_storage_numeric<T>()->value(0);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::ConstMapType>::type
    matrix() const {
        return matrix_storage_numeric<T>()->value(0);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, Eigen::Tensor<std::string, 2>&>::type
    matrix() {
        return matrix_storage_string()->value(0);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 2>&>::type
    matrix() const {
        return matrix_storage_string()->value(0);
    }

    template <typename T>
    void set_scalar(const T& v) {
        scalar<T>() = v;
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, void>::type
    set_vector(const typename NumericVectorTypes<T>::OwnedType& v) {
        vector_storage_numeric<T>()->set(0, v);
    }

    void set_vector(const Eigen::Tensor<std::string, 1>& v) {
        vector_storage_string()->value(0) = v;
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, void>::type
    set_matrix(const typename NumericMatrixTypes<T>::OwnedType& m) {
        matrix_storage_numeric<T>()->set(0, m);
    }

    void set_matrix(const Eigen::Tensor<std::string, 2>& m) {
        matrix_storage_string()->value(0) = m;
    }

private:
    template <typename T>
    ScalarStorage<T>* scalar_storage() {
        if (!storage_ || kind_ != CellKind::kScalar || dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        return static_cast<ScalarStorage<T>*>(storage_.get());
    }

    template <typename T>
    const ScalarStorage<T>* scalar_storage() const {
        if (!storage_ || kind_ != CellKind::kScalar || dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        return static_cast<const ScalarStorage<T>*>(storage_.get());
    }

    template <typename T>
    VectorStorageNumeric<T>* vector_storage_numeric() {
        if (!storage_ || kind_ != CellKind::kVector || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) {
            throw std::bad_cast();
        }
        return static_cast<VectorStorageNumeric<T>*>(storage_.get());
    }

    template <typename T>
    const VectorStorageNumeric<T>* vector_storage_numeric() const {
        if (!storage_ || kind_ != CellKind::kVector || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) {
            throw std::bad_cast();
        }
        return static_cast<const VectorStorageNumeric<T>*>(storage_.get());
    }

    VectorStorageString* vector_storage_string() {
        if (!storage_ || kind_ != CellKind::kVector || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<VectorStorageString*>(storage_.get());
    }

    const VectorStorageString* vector_storage_string() const {
        if (!storage_ || kind_ != CellKind::kVector || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<const VectorStorageString*>(storage_.get());
    }

    template <typename T>
    MatrixStorageNumeric<T>* matrix_storage_numeric() {
        if (!storage_ || kind_ != CellKind::kMatrix || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) {
            throw std::bad_cast();
        }
        return static_cast<MatrixStorageNumeric<T>*>(storage_.get());
    }

    template <typename T>
    const MatrixStorageNumeric<T>* matrix_storage_numeric() const {
        if (!storage_ || kind_ != CellKind::kMatrix || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) {
            throw std::bad_cast();
        }
        return static_cast<const MatrixStorageNumeric<T>*>(storage_.get());
    }

    MatrixStorageString* matrix_storage_string() {
        if (!storage_ || kind_ != CellKind::kMatrix || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<MatrixStorageString*>(storage_.get());
    }

    const MatrixStorageString* matrix_storage_string() const {
        if (!storage_ || kind_ != CellKind::kMatrix || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<const MatrixStorageString*>(storage_.get());
    }

    CellKind kind_;
    DTypeTag dtype_;
    std::vector<Index> shape_;
    std::unique_ptr<CellStorage> storage_;
};

class CellSeries {
public:
    struct ContiguousBlockView {
        const void* data;
        std::size_t elements;
        std::size_t bytes;
        std::size_t element_size;
        std::size_t rows;
        CellKind kind;
        DTypeTag dtype;
        std::vector<Index> cell_shape;
        bool trivially_copyable;
    };

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
        iterator(CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}

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
        CellSeries* owner_;
        std::size_t idx_;
    };

    class const_iterator {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef ConstRowView value_type;
        typedef std::ptrdiff_t difference_type;
        typedef void pointer;
        typedef ConstRowView reference;

        const_iterator() : owner_(nullptr), idx_(0) {}
        const_iterator(const CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}

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
        const CellSeries* owner_;
        std::size_t idx_;
    };

    CellSeries()
        : kind_(CellKind::kScalar),
          dtype_(DTypeTag::kReal),
          shape_(),
          storage_(make_storage(CellKind::kScalar, DTypeTag::kReal, std::vector<Index>())) {}

    CellSeries(CellKind kind, DTypeTag dtype, const std::vector<Index>& shape)
        : kind_(kind), dtype_(dtype), shape_(shape), storage_(make_storage(kind, dtype, shape)) {}

    CellSeries(const CellSeries& other)
        : kind_(other.kind_), dtype_(other.dtype_), shape_(other.shape_), storage_(other.storage_->clone()) {}

    CellSeries& operator=(const CellSeries& other) {
        if (this != &other) {
            kind_ = other.kind_;
            dtype_ = other.dtype_;
            shape_ = other.shape_;
            storage_ = other.storage_->clone();
        }
        return *this;
    }

    CellSeries(CellSeries&& other) noexcept
        : kind_(other.kind_), dtype_(other.dtype_), shape_(std::move(other.shape_)), storage_(std::move(other.storage_)) {}

    CellSeries& operator=(CellSeries&& other) noexcept {
        kind_ = other.kind_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        storage_ = std::move(other.storage_);
        return *this;
    }

    template <typename T>
    static CellSeries Scalars(std::size_t rows, const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s(CellKind::kScalar, DTypeOf<T>::tag, std::vector<Index>());
        s.resize(rows);
        s.fill(fill_val);
        return s;
    }

    template <typename T>
    static CellSeries ScalarsFrom(const std::vector<T>& values) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s(CellKind::kScalar, DTypeOf<T>::tag, std::vector<Index>());
        s.resize(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) s.scalar_at<T>(i) = values[i];
        return s;
    }

    template <typename T>
    static CellSeries ScalarsFrom(const T* values, std::size_t len) {
        static_assert(IsSupported<T>::value, "unsupported type");
        if (len > 0 && values == nullptr) {
            throw std::invalid_argument("values pointer must not be null when len > 0");
        }

        CellSeries s(CellKind::kScalar, DTypeOf<T>::tag, std::vector<Index>());
        s.resize(len);
        for (std::size_t i = 0; i < len; ++i) {
            s.scalar_at<T>(i) = values[i];
        }
        return s;
    }

    template <typename T>
    static CellSeries Vectors(std::size_t rows, Index width, const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s(CellKind::kVector, DTypeOf<T>::tag, std::vector<Index>(1, width));
        s.resize(rows);
        s.fill(fill_val);
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, CellSeries>::type
    VectorsFromFlat(Index width, const T* values, std::size_t len) {
        if (width < 0) {
            throw std::invalid_argument("vector width must be non-negative");
        }
        if (len > 0 && values == nullptr) {
            throw std::invalid_argument("values pointer must not be null when len > 0");
        }

        const std::size_t width_sz = static_cast<std::size_t>(width);
        if (width_sz == 0) {
            if (len != 0) {
                throw std::invalid_argument("vector width 0 requires len = 0");
            }
            return Vectors<T>(0, width);
        }
        if (len % width_sz != 0) {
            throw std::invalid_argument("vector flat data length must be a multiple of width");
        }

        const std::size_t rows = len / width_sz;
        CellSeries s = Vectors<T>(rows, width);
        T* dst = s.mutable_contiguous_data<T>();
        std::copy(values, values + len, dst);
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, CellSeries>::type
    VectorsFromFlat(Index width, const std::vector<T>& values) {
        return VectorsFromFlat<T>(
            width,
            values.empty() ? static_cast<const T*>(nullptr) : &values[0],
            values.size());
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, CellSeries>::type
    VectorsFromRows(const std::vector<typename NumericVectorTypes<T>::OwnedType>& rows) {
        if (rows.empty()) {
            return Vectors<T>(0, 0);
        }

        const Index width = static_cast<Index>(rows[0].size());
        CellSeries out = Vectors<T>(rows.size(), width);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].size() != width) {
                throw std::invalid_argument("all vector rows must have the same width");
            }
            out.vector_at<T>(i) = rows[i];
        }
        return out;
    }

    static CellSeries VectorsFromRows(const std::vector<Eigen::Tensor<std::string, 1> >& rows) {
        if (rows.empty()) {
            return Vectors<std::string>(0, 0);
        }

        const Index width = rows[0].dimension(0);
        CellSeries out = Vectors<std::string>(rows.size(), width);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].dimension(0) != width) {
                throw std::invalid_argument("all vector rows must have the same width");
            }
            out.vector_at<std::string>(i) = rows[i];
        }
        return out;
    }

    template <typename T>
    static CellSeries VectorsWithZeroRows(Index width) {
        return Vectors<T>(0, width);
    }

    template <typename T>
    static CellSeries Matrices(std::size_t rows, Index cell_rows, Index cell_cols, const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        std::vector<Index> shape;
        shape.push_back(cell_rows);
        shape.push_back(cell_cols);
        CellSeries s(CellKind::kMatrix, DTypeOf<T>::tag, shape);
        s.resize(rows);
        s.fill(fill_val);
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, CellSeries>::type
    MatricesFromFlat(Index cell_rows, Index cell_cols, const T* values, std::size_t len) {
        if (cell_rows < 0 || cell_cols < 0) {
            throw std::invalid_argument("matrix shape must be non-negative");
        }
        if (len > 0 && values == nullptr) {
            throw std::invalid_argument("values pointer must not be null when len > 0");
        }

        const std::size_t elems_per_row = static_cast<std::size_t>(cell_rows) *
                                          static_cast<std::size_t>(cell_cols);
        if (elems_per_row == 0) {
            if (len != 0) {
                throw std::invalid_argument("zero-sized matrix cells require len = 0");
            }
            return Matrices<T>(0, cell_rows, cell_cols);
        }
        if (len % elems_per_row != 0) {
            throw std::invalid_argument(
                "matrix flat data length must be a multiple of cell_rows * cell_cols");
        }

        const std::size_t rows = len / elems_per_row;
        CellSeries s = Matrices<T>(rows, cell_rows, cell_cols);
        T* dst = s.mutable_contiguous_data<T>();
        std::copy(values, values + len, dst);
        return s;
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, CellSeries>::type
    MatricesFromFlat(Index cell_rows, Index cell_cols, const std::vector<T>& values) {
        return MatricesFromFlat<T>(
            cell_rows,
            cell_cols,
            values.empty() ? static_cast<const T*>(nullptr) : &values[0],
            values.size());
    }

    template <typename T>
    static typename std::enable_if<!std::is_same<T, std::string>::value, CellSeries>::type
    MatricesFromRows(const std::vector<typename NumericMatrixTypes<T>::OwnedType>& rows) {
        if (rows.empty()) {
            return Matrices<T>(0, 0, 0);
        }

        const Index cell_rows = static_cast<Index>(rows[0].rows());
        const Index cell_cols = static_cast<Index>(rows[0].cols());
        CellSeries out = Matrices<T>(rows.size(), cell_rows, cell_cols);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].rows() != cell_rows || rows[i].cols() != cell_cols) {
                throw std::invalid_argument("all matrix rows must have the same shape");
            }
            out.matrix_at<T>(i) = rows[i];
        }
        return out;
    }

    static CellSeries MatricesFromRows(const std::vector<Eigen::Tensor<std::string, 2> >& rows) {
        if (rows.empty()) {
            return Matrices<std::string>(0, 0, 0);
        }

        const Index cell_rows = rows[0].dimension(0);
        const Index cell_cols = rows[0].dimension(1);
        CellSeries out = Matrices<std::string>(rows.size(), cell_rows, cell_cols);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].dimension(0) != cell_rows || rows[i].dimension(1) != cell_cols) {
                throw std::invalid_argument("all matrix rows must have the same shape");
            }
            out.matrix_at<std::string>(i) = rows[i];
        }
        return out;
    }

    template <typename T>
    static CellSeries MatricesWithZeroRows(Index cell_rows, Index cell_cols) {
        return Matrices<T>(0, cell_rows, cell_cols);
    }

    bool has_value() const { return storage_->size() != 0; }
    std::size_t size() const { return storage_->size(); }
    bool empty() const { return size() == 0; }

    CellKind cell_kind() const { return kind_; }
    DTypeTag dtype() const { return dtype_; }
    std::vector<Index> cell_shape() const { return shape_; }

    Index values_per_row() const {
        if (kind_ == CellKind::kScalar) return 1;
        if (kind_ == CellKind::kVector) return shape_[0];
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

    CellSeries head(std::size_t n) const {
        std::size_t sz = size();
        return iloc(0, n < sz ? n : sz);
    }

    CellSeries tail(std::size_t n) const {
        std::size_t sz = size();
        return iloc(n < sz ? sz - n : 0, sz);
    }

    CellSeries iloc(std::size_t start, std::size_t end) const {
        if (start > end || end > size()) throw std::out_of_range("iloc out of range");
        CellSeries out(kind_, dtype_, shape_);
        for (std::size_t i = start; i < end; ++i) out.append_from(*this, i);
        return out;
    }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size()); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, size()); }
    const_iterator cbegin() const { return const_iterator(this, 0); }
    const_iterator cend() const { return const_iterator(this, size()); }

    RowView row(std::size_t i);
    ConstRowView row(std::size_t i) const;

    template <typename T>
    T& scalar_at(std::size_t i) {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return scalar_storage<T>()->value(i);
    }

    template <typename T>
    const T& scalar_at(std::size_t i) const {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return scalar_storage<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::MapType>::type
    vector_at(std::size_t i) {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return vector_storage_numeric<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericVectorTypes<T>::ConstMapType>::type
    vector_at(std::size_t i) const {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return vector_storage_numeric<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, Eigen::Tensor<std::string, 1>&>::type
    vector_at(std::size_t i) {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return vector_storage_string()->value(i);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 1>&>::type
    vector_at(std::size_t i) const {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return vector_storage_string()->value(i);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::MapType>::type
    matrix_at(std::size_t i) {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return matrix_storage_numeric<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value,
                            typename NumericMatrixTypes<T>::ConstMapType>::type
    matrix_at(std::size_t i) const {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return matrix_storage_numeric<T>()->value(i);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, Eigen::Tensor<std::string, 2>&>::type
    matrix_at(std::size_t i) {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return matrix_storage_string()->value(i);
    }

    template <typename T>
    typename std::enable_if<std::is_same<T, std::string>::value, const Eigen::Tensor<std::string, 2>&>::type
    matrix_at(std::size_t i) const {
        if (i >= size()) throw std::out_of_range("row index out of range");
        return matrix_storage_string()->value(i);
    }

    template <typename T>
    T& vector_elem(std::size_t row, Index i) {
        typename NumericVectorTypes<T>::MapType v = vector_at<T>(row);
        if (i < 0 || i >= v.size()) throw std::out_of_range("vector index out of range");
        return v(i);
    }

    template <typename T>
    const T vector_elem(std::size_t row, Index i) const {
        typename NumericVectorTypes<T>::ConstMapType v = vector_at<T>(row);
        if (i < 0 || i >= v.size()) throw std::out_of_range("vector index out of range");
        return v(i);
    }

    std::string& vector_elem_string(std::size_t row, Index i) {
        Eigen::Tensor<std::string, 1>& v = vector_at<std::string>(row);
        if (i < 0 || i >= v.dimension(0)) throw std::out_of_range("vector index out of range");
        return v(i);
    }

    const std::string& vector_elem_string(std::size_t row, Index i) const {
        const Eigen::Tensor<std::string, 1>& v = vector_at<std::string>(row);
        if (i < 0 || i >= v.dimension(0)) throw std::out_of_range("vector index out of range");
        return v(i);
    }

    template <typename T>
    T& matrix_elem(std::size_t row, Index r, Index c) {
        typename NumericMatrixTypes<T>::MapType m = matrix_at<T>(row);
        if (r < 0 || c < 0 || r >= m.rows() || c >= m.cols()) throw std::out_of_range("matrix index out of range");
        return m(r, c);
    }

    template <typename T>
    const T matrix_elem(std::size_t row, Index r, Index c) const {
        typename NumericMatrixTypes<T>::ConstMapType m = matrix_at<T>(row);
        if (r < 0 || c < 0 || r >= m.rows() || c >= m.cols()) throw std::out_of_range("matrix index out of range");
        return m(r, c);
    }

    std::string& matrix_elem_string(std::size_t row, Index r, Index c) {
        Eigen::Tensor<std::string, 2>& m = matrix_at<std::string>(row);
        if (r < 0 || c < 0 || r >= m.dimension(0) || c >= m.dimension(1)) throw std::out_of_range("matrix index out of range");
        return m(r, c);
    }

    const std::string& matrix_elem_string(std::size_t row, Index r, Index c) const {
        const Eigen::Tensor<std::string, 2>& m = matrix_at<std::string>(row);
        if (r < 0 || c < 0 || r >= m.dimension(0) || c >= m.dimension(1)) throw std::out_of_range("matrix index out of range");
        return m(r, c);
    }

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

    void append_from(const CellSeries& src, std::size_t row) {
        if (src.kind_ != kind_ || src.dtype_ != dtype_ || src.shape_ != shape_) throw std::bad_cast();
        if (row >= src.size()) throw std::out_of_range("row index out of range");

        if (kind_ == CellKind::kScalar) {
            if (dtype_ == DTypeTag::kReal) append_scalar<double>(src.scalar_at<double>(row));
            else if (dtype_ == DTypeTag::kInteger) append_scalar<int>(src.scalar_at<int>(row));
            else if (dtype_ == DTypeTag::kComplex) append_scalar<std::complex<double> >(src.scalar_at<std::complex<double> >(row));
            else append_scalar<std::string>(src.scalar_at<std::string>(row));
            return;
        }

        if (kind_ == CellKind::kVector) {
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

    void assign_from(const CellSeries& src, std::size_t src_row, std::size_t dst_row) {
        if (src.kind_ != kind_ || src.dtype_ != dtype_ || src.shape_ != shape_) throw std::bad_cast();
        if (src_row >= src.size() || dst_row >= size()) throw std::out_of_range("row index out of range");

        if (kind_ == CellKind::kScalar) {
            if (dtype_ == DTypeTag::kReal) scalar_at<double>(dst_row) = src.scalar_at<double>(src_row);
            else if (dtype_ == DTypeTag::kInteger) scalar_at<int>(dst_row) = src.scalar_at<int>(src_row);
            else if (dtype_ == DTypeTag::kComplex) {
                scalar_at<std::complex<double> >(dst_row) = src.scalar_at<std::complex<double> >(src_row);
            } else {
                scalar_at<std::string>(dst_row) = src.scalar_at<std::string>(src_row);
            }
            return;
        }

        if (kind_ == CellKind::kVector) {
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

    void append(const Cell& cell) {
        if (!cell.has_value()) throw std::runtime_error("Cell has no value");
        if (cell.kind() != kind_ || cell.dtype() != dtype_ || cell.cell_shape() != shape_) throw std::bad_cast();

        if (kind_ == CellKind::kScalar) {
            if (dtype_ == DTypeTag::kReal) append_scalar<double>(cell.scalar<double>());
            else if (dtype_ == DTypeTag::kInteger) append_scalar<int>(cell.scalar<int>());
            else if (dtype_ == DTypeTag::kComplex) append_scalar<std::complex<double> >(cell.scalar<std::complex<double> >());
            else append_scalar<std::string>(cell.scalar<std::string>());
            return;
        }

        if (kind_ == CellKind::kVector) {
            if (dtype_ == DTypeTag::kReal) append_vector<double>(cell.vector<double>());
            else if (dtype_ == DTypeTag::kInteger) append_vector<int>(cell.vector<int>());
            else if (dtype_ == DTypeTag::kComplex) append_vector<std::complex<double> >(cell.vector<std::complex<double> >());
            else append_vector(cell.vector<std::string>());
            return;
        }

        if (dtype_ == DTypeTag::kReal) append_matrix<double>(cell.matrix<double>());
        else if (dtype_ == DTypeTag::kInteger) append_matrix<int>(cell.matrix<int>());
        else if (dtype_ == DTypeTag::kComplex) append_matrix<std::complex<double> >(cell.matrix<std::complex<double> >());
        else append_matrix(cell.matrix<std::string>());
    }

    template <typename T>
    void fill(const T& val) {
        if (dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        for (std::size_t i = 0; i < size(); ++i) {
            if (kind_ == CellKind::kScalar) {
                scalar_at<T>(i) = val;
            } else if (kind_ == CellKind::kVector) {
                fill_vector_row(i, val, std::is_same<T, std::string>());
            } else {
                fill_matrix_row(i, val, std::is_same<T, std::string>());
            }
        }
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, T*>::type mutable_contiguous_data() {
        if (dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        if (kind_ == CellKind::kScalar) return scalar_storage<T>()->data();
        if (kind_ == CellKind::kVector) return vector_storage_numeric<T>()->data();
        return matrix_storage_numeric<T>()->data();
    }

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, const T*>::type contiguous_data() const {
        if (dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        if (kind_ == CellKind::kScalar) return scalar_storage<T>()->data();
        if (kind_ == CellKind::kVector) return vector_storage_numeric<T>()->data();
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

    template <typename T>
    typename std::enable_if<!std::is_same<T, std::string>::value, ContiguousBlockView>::type
    export_contiguous_view() const {
        if (dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        ContiguousBlockView out;
        out.data = contiguous_data<T>();
        out.elements = contiguous_elements();
        out.bytes = out.elements * sizeof(T);
        out.element_size = sizeof(T);
        out.rows = size();
        out.kind = kind_;
        out.dtype = dtype_;
        out.cell_shape = shape_;
        out.trivially_copyable = true;
        return out;
    }

    ContiguousBlockView export_contiguous_view() const {
        if (dtype_ == DTypeTag::kReal) return export_contiguous_view<double>();
        if (dtype_ == DTypeTag::kInteger) return export_contiguous_view<int>();
        if (dtype_ == DTypeTag::kComplex) return export_contiguous_view<std::complex<double> >();
        throw std::runtime_error("string storage is not trivially-copyable");
    }

    Cell cell_at(std::size_t i) const {
        if (i >= size()) throw std::out_of_range("cell index out of range");

        if (kind_ == CellKind::kScalar) {
            if (dtype_ == DTypeTag::kReal) return Cell::Scalar<double>(scalar_at<double>(i));
            if (dtype_ == DTypeTag::kInteger) return Cell::Scalar<int>(scalar_at<int>(i));
            if (dtype_ == DTypeTag::kComplex) return Cell::Scalar<std::complex<double> >(scalar_at<std::complex<double> >(i));
            return Cell::Scalar<std::string>(scalar_at<std::string>(i));
        }

        if (kind_ == CellKind::kVector) {
            if (dtype_ == DTypeTag::kReal) {
                typename NumericVectorTypes<double>::OwnedType v = vector_at<double>(i);
                return Cell::Vector<double>(v);
            }
            if (dtype_ == DTypeTag::kInteger) {
                typename NumericVectorTypes<int>::OwnedType v = vector_at<int>(i);
                return Cell::Vector<int>(v);
            }
            if (dtype_ == DTypeTag::kComplex) {
                typename NumericVectorTypes<std::complex<double> >::OwnedType v = vector_at<std::complex<double> >(i);
                return Cell::Vector<std::complex<double> >(v);
            }
            return Cell::Vector(vector_at<std::string>(i));
        }

        if (dtype_ == DTypeTag::kReal) {
            typename NumericMatrixTypes<double>::OwnedType m = matrix_at<double>(i);
            return Cell::Matrix<double>(m);
        }
        if (dtype_ == DTypeTag::kInteger) {
            typename NumericMatrixTypes<int>::OwnedType m = matrix_at<int>(i);
            return Cell::Matrix<int>(m);
        }
        if (dtype_ == DTypeTag::kComplex) {
            typename NumericMatrixTypes<std::complex<double> >::OwnedType m = matrix_at<std::complex<double> >(i);
            return Cell::Matrix<std::complex<double> >(m);
        }
        return Cell::Matrix(matrix_at<std::string>(i));
    }

    CellSeries at(const std::vector<Index>& selected, bool reduce_to_scalar = false) const {
        if (kind_ == CellKind::kScalar) {
            throw std::logic_error("at is invalid for scalar data");
        }

        if (kind_ == CellKind::kVector) {
            if (reduce_to_scalar) {
                if (selected.size() != 1) {
                    throw std::invalid_argument("scalar vector at requires exactly one index");
                }

                if (dtype_ == DTypeTag::kReal) {
                    CellSeries out = CellSeries::Scalars<double>(size());
                    for (std::size_t row = 0; row < size(); ++row) {
                        out.scalar_at<double>(row) = vector_elem<double>(row, selected[0]);
                    }
                    return out;
                }
                if (dtype_ == DTypeTag::kInteger) {
                    CellSeries out = CellSeries::Scalars<int>(size());
                    for (std::size_t row = 0; row < size(); ++row) {
                        out.scalar_at<int>(row) = vector_elem<int>(row, selected[0]);
                    }
                    return out;
                }
                if (dtype_ == DTypeTag::kComplex) {
                    CellSeries out = CellSeries::Scalars<std::complex<double> >(size());
                    for (std::size_t row = 0; row < size(); ++row) {
                        out.scalar_at<std::complex<double> >(row) =
                            vector_elem<std::complex<double> >(row, selected[0]);
                    }
                    return out;
                }

                CellSeries out = CellSeries::Scalars<std::string>(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    out.scalar_at<std::string>(row) = vector_elem_string(row, selected[0]);
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

    CellSeries at(
        const std::vector<Index>& selected_rows,
        const std::vector<Index>& selected_cols,
        bool row_reduce = false,
        bool col_reduce = false) const {
        if (kind_ == CellKind::kScalar) {
            throw std::logic_error("at is invalid for scalar data");
        }

        if (kind_ == CellKind::kVector) {
            throw std::invalid_argument("matrix at requires matrix data");
        }

        if (row_reduce && col_reduce) {
            if (selected_rows.size() != 1 || selected_cols.size() != 1) {
                throw std::invalid_argument("scalar matrix at requires exactly one row and one column index");
            }

            if (dtype_ == DTypeTag::kReal) {
                CellSeries out = CellSeries::Scalars<double>(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    out.scalar_at<double>(row) =
                        matrix_elem<double>(row, selected_rows[0], selected_cols[0]);
                }
                return out;
            }
            if (dtype_ == DTypeTag::kInteger) {
                CellSeries out = CellSeries::Scalars<int>(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    out.scalar_at<int>(row) =
                        matrix_elem<int>(row, selected_rows[0], selected_cols[0]);
                }
                return out;
            }
            if (dtype_ == DTypeTag::kComplex) {
                CellSeries out = CellSeries::Scalars<std::complex<double> >(size());
                for (std::size_t row = 0; row < size(); ++row) {
                    out.scalar_at<std::complex<double> >(row) =
                        matrix_elem<std::complex<double> >(row, selected_rows[0], selected_cols[0]);
                }
                return out;
            }

            CellSeries out = CellSeries::Scalars<std::string>(size());
            for (std::size_t row = 0; row < size(); ++row) {
                out.scalar_at<std::string>(row) =
                    matrix_elem_string(row, selected_rows[0], selected_cols[0]);
            }
            return out;
        }

        if (row_reduce || col_reduce) {
            const bool select_columns = row_reduce;
            const std::vector<Index>& remaining = select_columns ? selected_cols : selected_rows;

            if (dtype_ == DTypeTag::kReal) {
                CellSeries out = CellSeries::Vectors<double>(size(), static_cast<Index>(remaining.size()));
                for (std::size_t row = 0; row < size(); ++row) {
                    auto out_vec = out.vector_at<double>(row);
                    for (std::size_t i = 0; i < remaining.size(); ++i) {
                        const Index r = row_reduce ? selected_rows[0] : remaining[i];
                        const Index c = row_reduce ? remaining[i] : selected_cols[0];
                        out_vec(static_cast<Index>(i)) = matrix_elem<double>(row, r, c);
                    }
                }
                return out;
            }
            if (dtype_ == DTypeTag::kInteger) {
                CellSeries out = CellSeries::Vectors<int>(size(), static_cast<Index>(remaining.size()));
                for (std::size_t row = 0; row < size(); ++row) {
                    auto out_vec = out.vector_at<int>(row);
                    for (std::size_t i = 0; i < remaining.size(); ++i) {
                        const Index r = row_reduce ? selected_rows[0] : remaining[i];
                        const Index c = row_reduce ? remaining[i] : selected_cols[0];
                        out_vec(static_cast<Index>(i)) = matrix_elem<int>(row, r, c);
                    }
                }
                return out;
            }
            if (dtype_ == DTypeTag::kComplex) {
                CellSeries out = CellSeries::Vectors<std::complex<double> >(
                    size(), static_cast<Index>(remaining.size()));
                for (std::size_t row = 0; row < size(); ++row) {
                    auto out_vec = out.vector_at<std::complex<double> >(row);
                    for (std::size_t i = 0; i < remaining.size(); ++i) {
                        const Index r = row_reduce ? selected_rows[0] : remaining[i];
                        const Index c = row_reduce ? remaining[i] : selected_cols[0];
                        out_vec(static_cast<Index>(i)) = matrix_elem<std::complex<double> >(row, r, c);
                    }
                }
                return out;
            }

            CellSeries out = CellSeries::Vectors<std::string>(size(), static_cast<Index>(remaining.size()));
            for (std::size_t row = 0; row < size(); ++row) {
                auto& out_vec = out.vector_at<std::string>(row);
                for (std::size_t i = 0; i < remaining.size(); ++i) {
                    const Index r = row_reduce ? selected_rows[0] : remaining[i];
                    const Index c = row_reduce ? remaining[i] : selected_cols[0];
                    out_vec(static_cast<Index>(i)) = matrix_elem_string(row, r, c);
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
    CellSeries at_vector_numeric_impl(const std::vector<Index>& selected) const {
        CellSeries out = CellSeries::Vectors<T>(size(), static_cast<Index>(selected.size()));
        for (std::size_t row = 0; row < size(); ++row) {
            auto out_vec = out.vector_at<T>(row);
            for (std::size_t i = 0; i < selected.size(); ++i) {
                out_vec(static_cast<Index>(i)) = vector_elem<T>(row, selected[i]);
            }
        }
        return out;
    }

    CellSeries at_vector_string_impl(const std::vector<Index>& selected) const {
        CellSeries out = CellSeries::Vectors<std::string>(size(), static_cast<Index>(selected.size()));
        for (std::size_t row = 0; row < size(); ++row) {
            auto out_vec = out.vector_at<std::string>(row);
            for (std::size_t i = 0; i < selected.size(); ++i) {
                out_vec(static_cast<Index>(i)) = vector_elem_string(row, selected[i]);
            }
        }
        return out;
    }

    template <typename T>
    CellSeries at_matrix_numeric_impl(
        const std::vector<Index>& selected_rows,
        const std::vector<Index>& selected_cols) const {

        CellSeries out = CellSeries::Matrices<T>(
            size(),
            static_cast<Index>(selected_rows.size()),
            static_cast<Index>(selected_cols.size()));

        for (std::size_t row = 0; row < size(); ++row) {
            auto out_mat = out.matrix_at<T>(row);
            for (std::size_t r = 0; r < selected_rows.size(); ++r) {
                for (std::size_t c = 0; c < selected_cols.size(); ++c) {
                    out_mat(static_cast<Index>(r), static_cast<Index>(c)) =
                        matrix_elem<T>(row, selected_rows[r], selected_cols[c]);
                }
            }
        }
        return out;
    }

    CellSeries at_matrix_string_impl(
        const std::vector<Index>& selected_rows,
        const std::vector<Index>& selected_cols) const {

        CellSeries out = CellSeries::Matrices<std::string>(
            size(),
            static_cast<Index>(selected_rows.size()),
            static_cast<Index>(selected_cols.size()));

        for (std::size_t row = 0; row < size(); ++row) {
            auto& out_mat = out.matrix_at<std::string>(row);
            for (std::size_t r = 0; r < selected_rows.size(); ++r) {
                for (std::size_t c = 0; c < selected_cols.size(); ++c) {
                    out_mat(static_cast<Index>(r), static_cast<Index>(c)) =
                        matrix_elem_string(row, selected_rows[r], selected_cols[c]);
                }
            }
        }
        return out;
    }

    static void validate_schema(CellKind kind, const std::vector<Index>& shape) {
        if (kind == CellKind::kScalar) {
            if (!shape.empty()) throw std::invalid_argument("scalar schema must have empty shape");
            return;
        }

        if (kind == CellKind::kVector) {
            if (shape.size() != 1 || shape[0] < 0) {
                throw std::invalid_argument("vector schema must have one non-negative dimension");
            }
            return;
        }

        if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0) {
            throw std::invalid_argument("matrix schema must have two non-negative dimensions");
        }
    }

    static std::unique_ptr<CellStorage> make_storage(CellKind kind, DTypeTag dtype, const std::vector<Index>& shape) {
        validate_schema(kind, shape);
        if (kind == CellKind::kScalar) {
            if (dtype == DTypeTag::kReal) return std::unique_ptr<CellStorage>(new ScalarStorage<double>());
            if (dtype == DTypeTag::kInteger) return std::unique_ptr<CellStorage>(new ScalarStorage<int>());
            if (dtype == DTypeTag::kComplex) return std::unique_ptr<CellStorage>(new ScalarStorage<std::complex<double> >());
            return std::unique_ptr<CellStorage>(new ScalarStorage<std::string>());
        }

        if (kind == CellKind::kVector) {
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
        if (kind_ != CellKind::kScalar || dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        return static_cast<ScalarStorage<T>*>(storage_.get());
    }

    template <typename T>
    const ScalarStorage<T>* scalar_storage() const {
        if (kind_ != CellKind::kScalar || dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        return static_cast<const ScalarStorage<T>*>(storage_.get());
    }

    template <typename T>
    VectorStorageNumeric<T>* vector_storage_numeric() {
        if (kind_ != CellKind::kVector || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) throw std::bad_cast();
        return static_cast<VectorStorageNumeric<T>*>(storage_.get());
    }

    template <typename T>
    const VectorStorageNumeric<T>* vector_storage_numeric() const {
        if (kind_ != CellKind::kVector || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) throw std::bad_cast();
        return static_cast<const VectorStorageNumeric<T>*>(storage_.get());
    }

    VectorStorageString* vector_storage_string() {
        if (kind_ != CellKind::kVector || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<VectorStorageString*>(storage_.get());
    }

    const VectorStorageString* vector_storage_string() const {
        if (kind_ != CellKind::kVector || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<const VectorStorageString*>(storage_.get());
    }

    template <typename T>
    MatrixStorageNumeric<T>* matrix_storage_numeric() {
        if (kind_ != CellKind::kMatrix || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) throw std::bad_cast();
        return static_cast<MatrixStorageNumeric<T>*>(storage_.get());
    }

    template <typename T>
    const MatrixStorageNumeric<T>* matrix_storage_numeric() const {
        if (kind_ != CellKind::kMatrix || dtype_ != DTypeOf<T>::tag || std::is_same<T, std::string>::value) throw std::bad_cast();
        return static_cast<const MatrixStorageNumeric<T>*>(storage_.get());
    }

    MatrixStorageString* matrix_storage_string() {
        if (kind_ != CellKind::kMatrix || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<MatrixStorageString*>(storage_.get());
    }

    const MatrixStorageString* matrix_storage_string() const {
        if (kind_ != CellKind::kMatrix || dtype_ != DTypeTag::kString) throw std::bad_cast();
        return static_cast<const MatrixStorageString*>(storage_.get());
    }

    template <typename T>
    void fill_vector_row(std::size_t row, const T& val, std::false_type) {
        vector_at<T>(row).setConstant(val);
    }

    void fill_vector_row(std::size_t row, const std::string& val, std::true_type) {
        Eigen::Tensor<std::string, 1>& t = vector_at<std::string>(row);
        for (Index i = 0; i < t.dimension(0); ++i) t(i) = val;
    }

    template <typename T>
    void fill_matrix_row(std::size_t row, const T& val, std::false_type) {
        matrix_at<T>(row).setConstant(val);
    }

    void fill_matrix_row(std::size_t row, const std::string& val, std::true_type) {
        Eigen::Tensor<std::string, 2>& t = matrix_at<std::string>(row);
        for (Index r = 0; r < t.dimension(0); ++r) {
            for (Index c = 0; c < t.dimension(1); ++c) {
                t(r, c) = val;
            }
        }
    }

    CellKind kind_;
    DTypeTag dtype_;
    std::vector<Index> shape_;
    std::unique_ptr<CellStorage> storage_;
};

class CellSeries::RowView {
public:
    RowView(CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}

    bool has_value() const { return owner_ != nullptr && idx_ < owner_->size(); }
    CellKind kind() const { return checked_owner()->cell_kind(); }
    DTypeTag dtype() const { return checked_owner()->dtype(); }
    std::string dtype_name() const { return checked_owner()->dtype_name(); }
    std::vector<Index> cell_shape() const { return checked_owner()->cell_shape(); }
    std::size_t index() const { return idx_; }

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

    Cell to_cell() const { return checked_owner()->cell_at(idx_); }

private:
    CellSeries* checked_owner() const {
        if (!owner_ || idx_ >= owner_->size()) throw std::out_of_range("row view is invalid");
        return owner_;
    }

    CellSeries* owner_;
    std::size_t idx_;
};

class CellSeries::ConstRowView {
public:
    ConstRowView(const CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}

    bool has_value() const { return owner_ != nullptr && idx_ < owner_->size(); }
    CellKind kind() const { return checked_owner()->cell_kind(); }
    DTypeTag dtype() const { return checked_owner()->dtype(); }
    std::string dtype_name() const { return checked_owner()->dtype_name(); }
    std::vector<Index> cell_shape() const { return checked_owner()->cell_shape(); }
    std::size_t index() const { return idx_; }

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

    Cell to_cell() const { return checked_owner()->cell_at(idx_); }

private:
    const CellSeries* checked_owner() const {
        if (!owner_ || idx_ >= owner_->size()) throw std::out_of_range("row view is invalid");
        return owner_;
    }

    const CellSeries* owner_;
    std::size_t idx_;
};

inline CellSeries::RowView CellSeries::row(std::size_t i) {
    if (i >= size()) throw std::out_of_range("cell index out of range");
    return RowView(this, i);
}

inline CellSeries::ConstRowView CellSeries::row(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("cell index out of range");
    return ConstRowView(this, i);
}

inline CellSeries::iterator::reference CellSeries::iterator::operator*() const {
    return RowView(owner_, idx_);
}

inline CellSeries::const_iterator::reference CellSeries::const_iterator::operator*() const {
    return ConstRowView(owner_, idx_);
}

}  // namespace xdataset

#endif  // XDATASET_TENSOR_SERIES_H_
