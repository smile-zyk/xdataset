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

// ---------- Type traits ----------

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

// Vector cell row type: numeric -> Eigen::VectorX<T>; string -> Eigen::Tensor<string,1>
template <typename T>
struct VectorRowType {
    typedef Eigen::VectorX<T> type;
};
template <>
struct VectorRowType<std::string> {
    typedef Eigen::Tensor<std::string, 1> type;
};

// Matrix cell row type: numeric -> Eigen::MatrixX<T>; string -> Eigen::Tensor<string,2>
template <typename T>
struct MatrixRowType {
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> type;
};
template <>
struct MatrixRowType<std::string> {
    typedef Eigen::Tensor<std::string, 2> type;
};

// Forward declaration: CellSeries can append a standalone Cell.
class CellSeries;

// ============================================================================
// CellStorage -- abstract multi-row column storage.
// A standalone Cell owns a storage with exactly one row; a CellSeries owns a
// storage with many rows. The same block type therefore serves both.
// ============================================================================

class CellStorage {
public:
    virtual ~CellStorage() {}
    virtual std::size_t size() const = 0;
    virtual void resize(std::size_t rows) = 0;
    virtual std::unique_ptr<CellStorage> clone() const = 0;
};

template <typename T, CellKind Kind>
class CellStorageImpl : public CellStorage {
    static_assert(Kind == CellKind::kScalar || Kind == CellKind::kVector || Kind == CellKind::kMatrix,
                  "unsupported CellKind for CellStorageImpl");
};

template <typename T>
class CellStorageImpl<T, CellKind::kScalar> : public CellStorage {
    static_assert(IsSupported<T>::value, "T must be one of double, int, std::complex<double>, std::string");

public:
    CellStorageImpl() {}

    std::size_t size() const override { return values_.size(); }
    void resize(std::size_t rows) override { values_.resize(rows, T()); }
    std::unique_ptr<CellStorage> clone() const override {
        return std::unique_ptr<CellStorage>(new CellStorageImpl<T, CellKind::kScalar>(*this));
    }

    T& value(std::size_t row) { return values_[row]; }
    const T& value(std::size_t row) const { return values_[row]; }
    void append(const T& value) { values_.push_back(value); }

private:
    std::vector<T> values_;
};

template <typename T>
class CellStorageImpl<T, CellKind::kVector> : public CellStorage {
    static_assert(IsSupported<T>::value, "T must be one of double, int, std::complex<double>, std::string");

public:
    typedef typename VectorRowType<T>::type RowType;

    explicit CellStorageImpl(Index width) : width_(width) {}

    std::size_t size() const override { return rows_.size(); }
    void resize(std::size_t rows) override {
        if (rows <= rows_.size()) {
            rows_.resize(rows);
            return;
        }
        rows_.resize(rows, default_row());
    }
    std::unique_ptr<CellStorage> clone() const override {
        return std::unique_ptr<CellStorage>(new CellStorageImpl<T, CellKind::kVector>(*this));
    }

    RowType& value(std::size_t row) { return rows_[row]; }
    const RowType& value(std::size_t row) const { return rows_[row]; }
    void append(const RowType& row) { rows_.push_back(row); }

private:
    RowType default_row() const { return default_row(std::is_same<T, std::string>()); }
    RowType default_row(std::true_type) const {
        RowType r(width_);
        return r;
    }
    RowType default_row(std::false_type) const { return RowType::Zero(width_); }

    Index width_;
    std::vector<RowType> rows_;
};

template <typename T>
class CellStorageImpl<T, CellKind::kMatrix> : public CellStorage {
    static_assert(IsSupported<T>::value, "T must be one of double, int, std::complex<double>, std::string");

public:
    typedef typename MatrixRowType<T>::type RowType;

    CellStorageImpl(Index rows, Index cols) : cell_rows_(rows), cell_cols_(cols) {}

    std::size_t size() const override { return rows_.size(); }
    void resize(std::size_t rows) override {
        if (rows <= rows_.size()) {
            rows_.resize(rows);
            return;
        }
        rows_.resize(rows, default_row());
    }
    std::unique_ptr<CellStorage> clone() const override {
        return std::unique_ptr<CellStorage>(new CellStorageImpl<T, CellKind::kMatrix>(*this));
    }

    RowType& value(std::size_t row) { return rows_[row]; }
    const RowType& value(std::size_t row) const { return rows_[row]; }
    void append(const RowType& row) { rows_.push_back(row); }

private:
    RowType default_row() const { return default_row(std::is_same<T, std::string>()); }
    RowType default_row(std::true_type) const {
        RowType r(cell_rows_, cell_cols_);
        return r;
    }
    RowType default_row(std::false_type) const { return RowType::Zero(cell_rows_, cell_cols_); }

    Index cell_rows_;
    Index cell_cols_;
    std::vector<RowType> rows_;
};

template <typename T, CellKind Kind>
struct CellValueTraits {
    static const bool kValid = false;
};

template <typename T>
struct CellValueTraits<T, CellKind::kScalar> {
    static const bool kValid = IsSupported<T>::value;
    static const CellKind kKind = CellKind::kScalar;
    static const DTypeTag kDType = DTypeOf<T>::tag;
    typedef T ValueType;
    typedef CellStorageImpl<T, CellKind::kScalar> StorageType;
};

template <typename T>
struct CellValueTraits<T, CellKind::kVector> {
    static const bool kValid = IsSupported<T>::value;
    static const CellKind kKind = CellKind::kVector;
    static const DTypeTag kDType = DTypeOf<T>::tag;
    typedef typename VectorRowType<T>::type ValueType;
    typedef CellStorageImpl<T, CellKind::kVector> StorageType;
};

template <typename T>
struct CellValueTraits<T, CellKind::kMatrix> {
    static const bool kValid = IsSupported<T>::value;
    static const CellKind kKind = CellKind::kMatrix;
    static const DTypeTag kDType = DTypeOf<T>::tag;
    typedef typename MatrixRowType<T>::type ValueType;
    typedef CellStorageImpl<T, CellKind::kMatrix> StorageType;
};

// ============================================================================
// Cell -- an independent, type-erased single value (scalar / vector / matrix
// of one of the supported dtypes). It owns its own storage and can be held and
// used on its own, or appended into a CellSeries.
// ============================================================================

class Cell {
public:
    Cell()
        : kind_(CellKind::kScalar),
          dtype_(DTypeTag::kReal),
          shape_(),
          storage_(new CellStorageImpl<double, CellKind::kScalar>()) {
        static_cast<CellStorageImpl<double, CellKind::kScalar>*>(storage_.get())->resize(1);
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

    // ---- Factories (build a standalone Cell) ----

    template <typename T>
    static Cell Scalar(const T& value) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Cell c;
        c.kind_ = CellKind::kScalar;
        c.dtype_ = DTypeOf<T>::tag;
        CellStorageImpl<T, CellKind::kScalar>* impl = new CellStorageImpl<T, CellKind::kScalar>();
        c.storage_.reset(impl);
        impl->resize(1);
        impl->value(0) = value;
        return c;
    }

    template <typename T>
    static Cell Scalar() {
        return Scalar<T>(T());
    }

    template <typename T>
    static Cell Vector(const typename VectorRowType<T>::type& row) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Cell c;
        c.kind_ = CellKind::kVector;
        c.dtype_ = DTypeOf<T>::tag;
        c.shape_ = std::vector<Index>(1, static_cast<Index>(row.size()));
        CellStorageImpl<T, CellKind::kVector>* impl =
            new CellStorageImpl<T, CellKind::kVector>(static_cast<Index>(row.size()));
        c.storage_.reset(impl);
        impl->resize(1);
        impl->value(0) = row;
        return c;
    }

    template <typename T>
    static Cell Vector(Index width) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Cell c;
        c.kind_ = CellKind::kVector;
        c.dtype_ = DTypeOf<T>::tag;
        c.shape_ = std::vector<Index>(1, width);
        CellStorageImpl<T, CellKind::kVector>* impl = new CellStorageImpl<T, CellKind::kVector>(width);
        c.storage_.reset(impl);
        impl->resize(1);
        return c;
    }

    template <typename T>
    static Cell Matrix(const typename MatrixRowType<T>::type& row) {
        static_assert(IsSupported<T>::value, "unsupported type");
        Index r = mat_rows(row);
        Index cc = mat_cols(row);
        Cell c;
        c.kind_ = CellKind::kMatrix;
        c.dtype_ = DTypeOf<T>::tag;
        c.shape_.push_back(r);
        c.shape_.push_back(cc);
        CellStorageImpl<T, CellKind::kMatrix>* impl = new CellStorageImpl<T, CellKind::kMatrix>(r, cc);
        c.storage_.reset(impl);
        impl->resize(1);
        impl->value(0) = row;
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
        CellStorageImpl<T, CellKind::kMatrix>* impl = new CellStorageImpl<T, CellKind::kMatrix>(rows, cols);
        c.storage_.reset(impl);
        impl->resize(1);
        return c;
    }

    // ---- Properties ----

    bool has_value() const { return storage_.get() != nullptr; }
    CellKind kind() const { return kind_; }
    DTypeTag dtype() const { return dtype_; }
    std::string dtype_name() const { return dtype_name_of(dtype_); }
    std::vector<Index> cell_shape() const { return shape_; }

    // ---- Typed value access ----

    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid,
                            typename CellValueTraits<T, Kind>::ValueType&>::type
    value() {
        return access<T, Kind>()->value(0);
    }
    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid,
                            const typename CellValueTraits<T, Kind>::ValueType&>::type
    value() const {
        return access<T, Kind>()->value(0);
    }

    template <typename T>
    T& scalar() {
        return value<T, CellKind::kScalar>();
    }
    template <typename T>
    const T& scalar() const {
        return value<T, CellKind::kScalar>();
    }

    template <typename T>
    typename VectorRowType<T>::type& vector() {
        return value<T, CellKind::kVector>();
    }
    template <typename T>
    const typename VectorRowType<T>::type& vector() const {
        return value<T, CellKind::kVector>();
    }

    template <typename T>
    typename MatrixRowType<T>::type& matrix() {
        return value<T, CellKind::kMatrix>();
    }
    template <typename T>
    const typename MatrixRowType<T>::type& matrix() const {
        return value<T, CellKind::kMatrix>();
    }

    template <typename T>
    typename VectorRowType<T>::type::Scalar& vector_at(Index i) {
        typename VectorRowType<T>::type& v = vector<T>();
        if (i < 0 || i >= static_cast<Index>(v.size())) throw std::out_of_range("vector index out of range");
        return v(i);
    }
    template <typename T>
    const typename VectorRowType<T>::type::Scalar& vector_at(Index i) const {
        const typename VectorRowType<T>::type& v = vector<T>();
        if (i < 0 || i >= static_cast<Index>(v.size())) throw std::out_of_range("vector index out of range");
        return v(i);
    }

    template <typename T>
    typename MatrixRowType<T>::type::Scalar& matrix_at(Index r, Index c) {
        typename MatrixRowType<T>::type& m = matrix<T>();
        if (r < 0 || c < 0 || r >= mat_rows(m) || c >= mat_cols(m)) {
            throw std::out_of_range("matrix index out of range");
        }
        return m(r, c);
    }
    template <typename T>
    const typename MatrixRowType<T>::type::Scalar& matrix_at(Index r, Index c) const {
        const typename MatrixRowType<T>::type& m = matrix<T>();
        if (r < 0 || c < 0 || r >= mat_rows(m) || c >= mat_cols(m)) {
            throw std::out_of_range("matrix index out of range");
        }
        return m(r, c);
    }

    // ---- Set ----

    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid, void>::type set_value(
        const typename CellValueTraits<T, Kind>::ValueType& v) {
        value<T, Kind>() = v;
    }

    template <typename T>
    void set_scalar(const T& v) {
        value<T, CellKind::kScalar>() = v;
    }
    template <typename T>
    void set_vector(const typename VectorRowType<T>::type& v) {
        value<T, CellKind::kVector>() = v;
    }
    template <typename T>
    void set_matrix(const typename MatrixRowType<T>::type& v) {
        value<T, CellKind::kMatrix>() = v;
    }
    template <typename T>
    void set_vector_at(Index i, const typename VectorRowType<T>::type::Scalar& v) {
        vector_at<T>(i) = v;
    }
    template <typename T>
    void set_matrix_at(Index r, Index c, const typename MatrixRowType<T>::type::Scalar& v) {
        matrix_at<T>(r, c) = v;
    }

    // ---- Small shared helpers (no free functions) ----

    static std::string dtype_name_of(DTypeTag tag) {
        switch (tag) {
            case DTypeTag::kReal: return "Real";
            case DTypeTag::kInteger: return "Integer";
            case DTypeTag::kComplex: return "Complex";
            case DTypeTag::kString: return "String";
        }
        return "Unknown";
    }

    static Index mat_rows(const Eigen::Tensor<std::string, 2>& m) { return m.dimension(0); }
    static Index mat_cols(const Eigen::Tensor<std::string, 2>& m) { return m.dimension(1); }
    template <typename M>
    static Index mat_rows(const M& m) {
        return static_cast<Index>(m.rows());
    }
    template <typename M>
    static Index mat_cols(const M& m) {
        return static_cast<Index>(m.cols());
    }

private:
    template <typename T, CellKind Kind>
    typename CellValueTraits<T, Kind>::StorageType* access() {
        if (!storage_) throw std::runtime_error("Cell has no value");
        typedef CellValueTraits<T, Kind> Traits;
        if (kind_ != Kind || dtype_ != Traits::kDType) throw std::bad_cast();
        return static_cast<typename Traits::StorageType*>(storage_.get());
    }
    template <typename T, CellKind Kind>
    const typename CellValueTraits<T, Kind>::StorageType* access() const {
        if (!storage_) throw std::runtime_error("Cell has no value");
        typedef CellValueTraits<T, Kind> Traits;
        if (kind_ != Kind || dtype_ != Traits::kDType) throw std::bad_cast();
        return static_cast<const typename Traits::StorageType*>(storage_.get());
    }

    CellKind kind_;
    DTypeTag dtype_;
    std::vector<Index> shape_;
    std::unique_ptr<CellStorage> storage_;
};

// ============================================================================
// CellSeries -- owns a whole column storage block and all row logic.
// Row slices / iteration are exposed through the nested RowView.
// ============================================================================

class CellSeries {
public:
    class RowView;  // slice/view over a single row (defined after Cell)
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

        reference operator*() const;  // defined out-of-line (needs RowView complete)

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

        reference operator*() const;  // defined out-of-line (needs RowView complete)

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
        : kind_(other.kind_),
          dtype_(other.dtype_),
          shape_(other.shape_),
          storage_(other.storage_->clone()) {}

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
        : kind_(other.kind_),
          dtype_(other.dtype_),
          shape_(std::move(other.shape_)),
          storage_(std::move(other.storage_)) {}

    CellSeries& operator=(CellSeries&& other) noexcept {
        kind_ = other.kind_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        storage_ = std::move(other.storage_);
        return *this;
    }

    // ---- Factory functions ----

    template <typename T>
    static CellSeries Scalars(std::size_t rows, const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s(CellKind::kScalar, DTypeOf<T>::tag, std::vector<Index>());
        s.resize(rows);
        s.fill(fill_val);
        return s;
    }

    template <typename T>
    static CellSeries From(const std::vector<T>& values) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s(CellKind::kScalar, DTypeOf<T>::tag, std::vector<Index>());
        s.resize(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) s.scalar_at<T>(i) = values[i];
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
    static CellSeries Matrices(std::size_t rows, Index cell_rows, Index cell_cols,
                               const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        std::vector<Index> shape;
        shape.push_back(cell_rows);
        shape.push_back(cell_cols);
        CellSeries s(CellKind::kMatrix, DTypeOf<T>::tag, shape);
        s.resize(rows);
        s.fill(fill_val);
        return s;
    }

    // ---- Properties ----

    bool has_value() const { return storage_->size() != 0; }
    std::size_t size() const { return storage_->size(); }
    bool empty() const { return size() == 0; }

    CellKind cell_kind() const { return kind_; }
    std::vector<Index> cell_shape() const { return shape_; }

    Index values_per_row() const {
        if (kind_ == CellKind::kScalar) return 1;
        if (kind_ == CellKind::kVector) return shape_[0];
        return shape_[0] * shape_[1];
    }

    std::string dtype_name() const { return Cell::dtype_name_of(dtype_); }
    DTypeTag dtype() const { return dtype_; }

    // ---- Resize / clear ----

    void resize(std::size_t n) { storage_->resize(n); }
    void clear() { storage_->resize(0); }

    // ---- Slicing ----

    CellSeries head(std::size_t n) const { return iloc(0, n); }

    CellSeries tail(std::size_t n) const {
        std::size_t sz = size();
        return iloc(sz > n ? sz - n : 0, sz);
    }

    CellSeries iloc(std::size_t start, std::size_t end) const {
        if (start > end || end > size()) throw std::out_of_range("iloc out of range");
        CellSeries out(kind_, dtype_, shape_);
        for (std::size_t i = start; i < end; ++i) out.append_from(*this, i);
        return out;
    }

    // Independent, owning copy of one row.
    Cell cell_at(std::size_t i) const;  // defined out-of-line

    // ---- Iterators ----

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size()); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, size()); }
    const_iterator cbegin() const { return const_iterator(this, 0); }
    const_iterator cend() const { return const_iterator(this, size()); }

    RowView row(std::size_t i);              // defined out-of-line
    ConstRowView row(std::size_t i) const;   // defined out-of-line

    // ---- Typed value access (with row parameter) ----

    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid,
                            typename CellValueTraits<T, Kind>::ValueType&>::type
    value(std::size_t row) {
        if (row >= size()) throw std::out_of_range("row index out of range");
        return typed_storage<T, Kind>()->value(row);
    }

    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid,
                            const typename CellValueTraits<T, Kind>::ValueType&>::type
    value(std::size_t row) const {
        if (row >= size()) throw std::out_of_range("row index out of range");
        return typed_storage<T, Kind>()->value(row);
    }

    template <typename T>
    T& scalar_at(std::size_t i) {
        return value<T, CellKind::kScalar>(i);
    }
    template <typename T>
    const T& scalar_at(std::size_t i) const {
        return value<T, CellKind::kScalar>(i);
    }

    template <typename T>
    typename VectorRowType<T>::type& vector_at(std::size_t i) {
        return value<T, CellKind::kVector>(i);
    }
    template <typename T>
    const typename VectorRowType<T>::type& vector_at(std::size_t i) const {
        return value<T, CellKind::kVector>(i);
    }

    template <typename T>
    typename MatrixRowType<T>::type& matrix_at(std::size_t i) {
        return value<T, CellKind::kMatrix>(i);
    }
    template <typename T>
    const typename MatrixRowType<T>::type& matrix_at(std::size_t i) const {
        return value<T, CellKind::kMatrix>(i);
    }

    // ---- Element access (with row parameter) ----

    template <typename T>
    typename VectorRowType<T>::type::Scalar& vector_elem(std::size_t row, Index i) {
        typename VectorRowType<T>::type& v = vector_at<T>(row);
        if (i < 0 || i >= static_cast<Index>(v.size())) throw std::out_of_range("vector index out of range");
        return v(i);
    }
    template <typename T>
    const typename VectorRowType<T>::type::Scalar& vector_elem(std::size_t row, Index i) const {
        const typename VectorRowType<T>::type& v = vector_at<T>(row);
        if (i < 0 || i >= static_cast<Index>(v.size())) throw std::out_of_range("vector index out of range");
        return v(i);
    }

    template <typename T>
    typename MatrixRowType<T>::type::Scalar& matrix_elem(std::size_t row, Index r, Index c) {
        typename MatrixRowType<T>::type& m = matrix_at<T>(row);
        if (r < 0 || c < 0 || r >= Cell::mat_rows(m) || c >= Cell::mat_cols(m)) {
            throw std::out_of_range("matrix index out of range");
        }
        return m(r, c);
    }
    template <typename T>
    const typename MatrixRowType<T>::type::Scalar& matrix_elem(std::size_t row, Index r, Index c) const {
        const typename MatrixRowType<T>::type& m = matrix_at<T>(row);
        if (r < 0 || c < 0 || r >= Cell::mat_rows(m) || c >= Cell::mat_cols(m)) {
            throw std::out_of_range("matrix index out of range");
        }
        return m(r, c);
    }

    // ---- Set (with row parameter) ----

    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid, void>::type
    set_value(std::size_t row, const typename CellValueTraits<T, Kind>::ValueType& v) {
        value<T, Kind>(row) = v;
    }

    // ---- Append ----

    void append(const Cell& cell);  // defined out-of-line

    void append_from(const CellSeries& src, std::size_t row) {
        if (src.kind_ != kind_ || src.dtype_ != dtype_ || src.shape_ != shape_) throw std::bad_cast();
        if (row >= src.size()) throw std::out_of_range("row index out of range");
        append_row_from(src, row);
    }

    template <typename T>
    typename std::enable_if<IsSupported<T>::value, void>::type append_scalar(const T& val) {
        typed_storage<T, CellKind::kScalar>()->append(val);
    }

    template <typename T>
    void append_vector(const typename VectorRowType<T>::type& v) {
        if (static_cast<Index>(v.size()) != values_per_row()) throw std::bad_cast();
        typed_storage<T, CellKind::kVector>()->append(v);
    }

    template <typename T>
    void append_matrix(const typename MatrixRowType<T>::type& m) {
        if (Cell::mat_rows(m) != shape_[0] || Cell::mat_cols(m) != shape_[1]) throw std::bad_cast();
        typed_storage<T, CellKind::kMatrix>()->append(m);
    }

    // ---- Fill ----

    template <typename T>
    void fill(const T& val) {
        if (dtype_ != DTypeOf<T>::tag) throw std::bad_cast();
        for (std::size_t i = 0; i < size(); ++i) {
            if (kind_ == CellKind::kScalar) {
                scalar_at<T>(i) = val;
            } else if (kind_ == CellKind::kVector) {
                typename VectorRowType<T>::type& v = vector_at<T>(i);
                fill_vector_row(v, val, std::is_same<T, std::string>());
            } else {
                typename MatrixRowType<T>::type& m = matrix_at<T>(i);
                fill_matrix_row(m, val, std::is_same<T, std::string>());
            }
        }
    }

private:
    template <typename RowType, typename TVal>
    static void fill_vector_row(RowType& row, const TVal& val, std::false_type) {
        row.setConstant(val);
    }

    static void fill_vector_row(Eigen::Tensor<std::string, 1>& row, const std::string& val,
                                std::true_type) {
        for (Index j = 0; j < row.dimension(0); ++j) row(j) = val;
    }

    template <typename RowType, typename TVal>
    static void fill_matrix_row(RowType& row, const TVal& val, std::false_type) {
        row.setConstant(val);
    }

    static void fill_matrix_row(Eigen::Tensor<std::string, 2>& row, const std::string& val,
                                std::true_type) {
        for (Index r = 0; r < row.dimension(0); ++r)
            for (Index c = 0; c < row.dimension(1); ++c)
                row(r, c) = val;
    }

    static std::unique_ptr<CellStorage> make_storage(
        CellKind kind, DTypeTag dtype, const std::vector<Index>& shape) {
        validate_schema(kind, shape);
        if (kind == CellKind::kScalar) {
            if (dtype == DTypeTag::kReal) return std::unique_ptr<CellStorage>(new CellStorageImpl<double, CellKind::kScalar>());
            if (dtype == DTypeTag::kInteger) return std::unique_ptr<CellStorage>(new CellStorageImpl<int, CellKind::kScalar>());
            if (dtype == DTypeTag::kComplex) return std::unique_ptr<CellStorage>(new CellStorageImpl<std::complex<double>, CellKind::kScalar>());
            return std::unique_ptr<CellStorage>(new CellStorageImpl<std::string, CellKind::kScalar>());
        }
        if (kind == CellKind::kVector) {
            if (dtype == DTypeTag::kReal) return std::unique_ptr<CellStorage>(new CellStorageImpl<double, CellKind::kVector>(shape[0]));
            if (dtype == DTypeTag::kInteger) return std::unique_ptr<CellStorage>(new CellStorageImpl<int, CellKind::kVector>(shape[0]));
            if (dtype == DTypeTag::kComplex) return std::unique_ptr<CellStorage>(new CellStorageImpl<std::complex<double>, CellKind::kVector>(shape[0]));
            return std::unique_ptr<CellStorage>(new CellStorageImpl<std::string, CellKind::kVector>(shape[0]));
        }
        if (dtype == DTypeTag::kReal) return std::unique_ptr<CellStorage>(new CellStorageImpl<double, CellKind::kMatrix>(shape[0], shape[1]));
        if (dtype == DTypeTag::kInteger) return std::unique_ptr<CellStorage>(new CellStorageImpl<int, CellKind::kMatrix>(shape[0], shape[1]));
        if (dtype == DTypeTag::kComplex) return std::unique_ptr<CellStorage>(new CellStorageImpl<std::complex<double>, CellKind::kMatrix>(shape[0], shape[1]));
        return std::unique_ptr<CellStorage>(new CellStorageImpl<std::string, CellKind::kMatrix>(shape[0], shape[1]));
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

    template <typename T, CellKind Kind>
    typename CellValueTraits<T, Kind>::StorageType* typed_storage() {
        typedef CellValueTraits<T, Kind> Traits;
        if (kind_ != Kind || dtype_ != Traits::kDType) throw std::bad_cast();
        return static_cast<typename Traits::StorageType*>(storage_.get());
    }

    template <typename T, CellKind Kind>
    const typename CellValueTraits<T, Kind>::StorageType* typed_storage() const {
        typedef CellValueTraits<T, Kind> Traits;
        if (kind_ != Kind || dtype_ != Traits::kDType) throw std::bad_cast();
        return static_cast<const typename Traits::StorageType*>(storage_.get());
    }

    template <typename T, CellKind Kind>
    void append_value_from(const CellSeries& src, std::size_t row) {
        typed_storage<T, Kind>()->append(src.value<T, Kind>(row));
    }

    template <CellKind Kind>
    void append_kind_from(const CellSeries& src, std::size_t row) {
        if (dtype_ == DTypeTag::kReal) append_value_from<double, Kind>(src, row);
        else if (dtype_ == DTypeTag::kInteger) append_value_from<int, Kind>(src, row);
        else if (dtype_ == DTypeTag::kComplex) append_value_from<std::complex<double>, Kind>(src, row);
        else append_value_from<std::string, Kind>(src, row);
    }

    void append_row_from(const CellSeries& src, std::size_t row) {
        if (kind_ == CellKind::kScalar) append_kind_from<CellKind::kScalar>(src, row);
        else if (kind_ == CellKind::kVector) append_kind_from<CellKind::kVector>(src, row);
        else append_kind_from<CellKind::kMatrix>(src, row);
    }

    // Cell -> series append helpers (defined out-of-line, need Cell complete).
    template <typename T, CellKind Kind>
    void append_cell_value(const Cell& cell);
    template <CellKind Kind>
    void append_cell_kind(const Cell& cell);
    void append_cell(const Cell& cell);

    CellKind kind_;
    DTypeTag dtype_;
    std::vector<Index> shape_;
    std::unique_ptr<CellStorage> storage_;
};

// ============================================================================
// CellSeries::RowView -- a single-row slice that depends on its owning series.
// It borrows the series' data (no ownership). If the series is destroyed the
// view is dangling; use to_cell() to obtain an independent Cell copy.
// ============================================================================

class CellSeries::RowView {
public:
    RowView(CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}

    bool has_value() const { return owner_ != nullptr && idx_ < owner_->size(); }
    CellKind kind() const { return checked_owner()->cell_kind(); }
    DTypeTag dtype() const { return checked_owner()->dtype(); }
    std::string dtype_name() const { return checked_owner()->dtype_name(); }
    std::vector<Index> cell_shape() const { return checked_owner()->cell_shape(); }
    std::size_t index() const { return idx_; }

    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid,
                            typename CellValueTraits<T, Kind>::ValueType&>::type
    value() {
        return checked_owner()->value<T, Kind>(idx_);
    }
    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid,
                            const typename CellValueTraits<T, Kind>::ValueType&>::type
    value() const {
        return checked_owner()->value<T, Kind>(idx_);
    }

    template <typename T>
    T& scalar() {
        return checked_owner()->scalar_at<T>(idx_);
    }
    template <typename T>
    const T& scalar() const {
        return checked_owner()->scalar_at<T>(idx_);
    }

    template <typename T>
    typename VectorRowType<T>::type& vector() {
        return checked_owner()->vector_at<T>(idx_);
    }
    template <typename T>
    const typename VectorRowType<T>::type& vector() const {
        return checked_owner()->vector_at<T>(idx_);
    }

    template <typename T>
    typename MatrixRowType<T>::type& matrix() {
        return checked_owner()->matrix_at<T>(idx_);
    }
    template <typename T>
    const typename MatrixRowType<T>::type& matrix() const {
        return checked_owner()->matrix_at<T>(idx_);
    }

    template <typename T>
    typename VectorRowType<T>::type::Scalar& vector_at(Index i) {
        return checked_owner()->vector_elem<T>(idx_, i);
    }
    template <typename T>
    const typename VectorRowType<T>::type::Scalar& vector_at(Index i) const {
        return checked_owner()->vector_elem<T>(idx_, i);
    }

    template <typename T>
    typename MatrixRowType<T>::type::Scalar& matrix_at(Index r, Index c) {
        return checked_owner()->matrix_elem<T>(idx_, r, c);
    }
    template <typename T>
    const typename MatrixRowType<T>::type::Scalar& matrix_at(Index r, Index c) const {
        return checked_owner()->matrix_elem<T>(idx_, r, c);
    }

    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid, void>::type set_value(
        const typename CellValueTraits<T, Kind>::ValueType& v) {
        checked_owner()->set_value<T, Kind>(idx_, v);
    }

    template <typename T>
    void set_scalar(const T& v) {
        checked_owner()->scalar_at<T>(idx_) = v;
    }
    template <typename T>
    void set_vector(const typename VectorRowType<T>::type& v) {
        checked_owner()->vector_at<T>(idx_) = v;
    }
    template <typename T>
    void set_matrix(const typename MatrixRowType<T>::type& v) {
        checked_owner()->matrix_at<T>(idx_) = v;
    }
    template <typename T>
    void set_vector_at(Index i, const typename VectorRowType<T>::type::Scalar& v) {
        checked_owner()->vector_elem<T>(idx_, i) = v;
    }
    template <typename T>
    void set_matrix_at(Index r, Index c, const typename MatrixRowType<T>::type::Scalar& v) {
        checked_owner()->matrix_elem<T>(idx_, r, c) = v;
    }

    // Materialize this view into an independent, owning Cell.
    Cell to_cell() const { return checked_owner()->cell_at(idx_); }

private:
    CellSeries* checked_owner() const {
        if (!owner_ || idx_ >= owner_->size()) {
            throw std::out_of_range("row view is invalid");
        }
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

    template <typename T, CellKind Kind>
    typename std::enable_if<CellValueTraits<T, Kind>::kValid,
                            const typename CellValueTraits<T, Kind>::ValueType&>::type
    value() const {
        return checked_owner()->value<T, Kind>(idx_);
    }

    template <typename T>
    const T& scalar() const {
        return checked_owner()->scalar_at<T>(idx_);
    }

    template <typename T>
    const typename VectorRowType<T>::type& vector() const {
        return checked_owner()->vector_at<T>(idx_);
    }

    template <typename T>
    const typename MatrixRowType<T>::type& matrix() const {
        return checked_owner()->matrix_at<T>(idx_);
    }

    template <typename T>
    const typename VectorRowType<T>::type::Scalar& vector_at(Index i) const {
        return checked_owner()->vector_elem<T>(idx_, i);
    }

    template <typename T>
    const typename MatrixRowType<T>::type::Scalar& matrix_at(Index r, Index c) const {
        return checked_owner()->matrix_elem<T>(idx_, r, c);
    }

    Cell to_cell() const { return checked_owner()->cell_at(idx_); }

private:
    const CellSeries* checked_owner() const {
        if (!owner_ || idx_ >= owner_->size()) {
            throw std::out_of_range("row view is invalid");
        }
        return owner_;
    }

    const CellSeries* owner_;
    std::size_t idx_;
};

// ---- Out-of-line CellSeries members that need RowView / Cell complete ----

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

inline Cell CellSeries::cell_at(std::size_t i) const {
    if (i >= size()) throw std::out_of_range("cell index out of range");
    if (kind_ == CellKind::kScalar) {
        if (dtype_ == DTypeTag::kReal) return Cell::Scalar<double>(value<double, CellKind::kScalar>(i));
        if (dtype_ == DTypeTag::kInteger) return Cell::Scalar<int>(value<int, CellKind::kScalar>(i));
        if (dtype_ == DTypeTag::kComplex) return Cell::Scalar<std::complex<double> >(value<std::complex<double>, CellKind::kScalar>(i));
        return Cell::Scalar<std::string>(value<std::string, CellKind::kScalar>(i));
    }
    if (kind_ == CellKind::kVector) {
        if (dtype_ == DTypeTag::kReal) return Cell::Vector<double>(value<double, CellKind::kVector>(i));
        if (dtype_ == DTypeTag::kInteger) return Cell::Vector<int>(value<int, CellKind::kVector>(i));
        if (dtype_ == DTypeTag::kComplex) return Cell::Vector<std::complex<double> >(value<std::complex<double>, CellKind::kVector>(i));
        return Cell::Vector<std::string>(value<std::string, CellKind::kVector>(i));
    }
    if (dtype_ == DTypeTag::kReal) return Cell::Matrix<double>(value<double, CellKind::kMatrix>(i));
    if (dtype_ == DTypeTag::kInteger) return Cell::Matrix<int>(value<int, CellKind::kMatrix>(i));
    if (dtype_ == DTypeTag::kComplex) return Cell::Matrix<std::complex<double> >(value<std::complex<double>, CellKind::kMatrix>(i));
    return Cell::Matrix<std::string>(value<std::string, CellKind::kMatrix>(i));
}

template <typename T, CellKind Kind>
inline void CellSeries::append_cell_value(const Cell& cell) {
    typed_storage<T, Kind>()->append(cell.value<T, Kind>());
}

template <CellKind Kind>
inline void CellSeries::append_cell_kind(const Cell& cell) {
    if (dtype_ == DTypeTag::kReal) append_cell_value<double, Kind>(cell);
    else if (dtype_ == DTypeTag::kInteger) append_cell_value<int, Kind>(cell);
    else if (dtype_ == DTypeTag::kComplex) append_cell_value<std::complex<double>, Kind>(cell);
    else append_cell_value<std::string, Kind>(cell);
}

inline void CellSeries::append_cell(const Cell& cell) {
    if (kind_ == CellKind::kScalar) append_cell_kind<CellKind::kScalar>(cell);
    else if (kind_ == CellKind::kVector) append_cell_kind<CellKind::kVector>(cell);
    else append_cell_kind<CellKind::kMatrix>(cell);
}

inline void CellSeries::append(const Cell& cell) {
    if (!cell.has_value()) throw std::runtime_error("Cell has no value");
    if (cell.kind() != kind_ || cell.dtype() != dtype_ || cell.cell_shape() != shape_) {
        throw std::bad_cast();
    }
    append_cell(cell);
}

}  // namespace xdataset

#endif  // XDATASET_TENSOR_SERIES_H_
