#ifndef XDATASET_TENSOR_SERIES_H_
#define XDATASET_TENSOR_SERIES_H_

#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace xdataset {

using Index = Eigen::Index;

// Cell shape kind
enum class CellKind { kScalar, kVector, kMatrix };

// ---------- Type traits ----------

template <typename T>
struct IsComplex : std::false_type {};
template <typename T>
struct IsComplex<std::complex<T> > : std::true_type {};

template <typename T>
struct IsNumeric
    : std::integral_constant<bool, std::is_arithmetic<T>::value || IsComplex<T>::value> {};

template <typename T>
struct IsSupported
    : std::integral_constant<bool, IsNumeric<T>::value || std::is_same<T, std::string>::value> {};

// Vector cell row type: numeric -> Eigen::VectorX<T>; string -> vector<string>
template <typename T>
struct VectorRowType {
    typedef Eigen::VectorX<T> type;
};
template <>
struct VectorRowType<std::string> {
    typedef std::vector<std::string> type;
};

// Matrix cell row type: numeric -> Eigen::MatrixX<T>; string -> vector<string> (row-major flat)
template <typename T>
struct MatrixRowType {
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> type;
};
template <>
struct MatrixRowType<std::string> {
    typedef std::vector<std::string> type;
};

// ---------- Abstract storage interface ----------

class SeriesStorage {
public:
    virtual ~SeriesStorage() {}
    virtual CellKind kind() const = 0;
    virtual const std::type_info& value_type() const = 0;
    virtual std::string dtype_name() const = 0;
    virtual std::size_t size() const = 0;
    virtual bool empty() const = 0;
    virtual std::vector<Index> cell_shape() const = 0;
    virtual Index values_per_row() const = 0;
    virtual void resize(std::size_t n) = 0;
    virtual void clear() = 0;
    virtual void append_from_row(const SeriesStorage& src, std::size_t src_index) = 0;
    virtual std::unique_ptr<SeriesStorage> clone() const = 0;
    virtual std::unique_ptr<SeriesStorage> slice(std::size_t start, std::size_t end) const = 0;
};

// ==========================================================================
// ScalarSeriesStorage<T>   storage: std::vector<T>
// ==========================================================================

template <typename T>
class ScalarSeriesStorage : public SeriesStorage {
    static_assert(IsSupported<T>::value, "T must be numeric or std::string");

public:
    ScalarSeriesStorage() {}
    explicit ScalarSeriesStorage(std::size_t n, const T& fill_val = T()) : data_(n, fill_val) {}
    explicit ScalarSeriesStorage(const std::vector<T>& data) : data_(data) {}

    CellKind kind() const { return CellKind::kScalar; }
    const std::type_info& value_type() const { return typeid(T); }
    std::string dtype_name() const { return typeid(T).name(); }
    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    std::vector<Index> cell_shape() const { return std::vector<Index>(); }
    Index values_per_row() const { return 1; }

    void resize(std::size_t n) { data_.resize(n); }
    void clear() { data_.clear(); }

    void append_from_row(const SeriesStorage& src, std::size_t src_index) {
        const ScalarSeriesStorage<T>* p = dynamic_cast<const ScalarSeriesStorage<T>*>(&src);
        if (!p) throw std::bad_cast();
        append(p->at(src_index));
    }

    std::unique_ptr<SeriesStorage> clone() const {
        return std::unique_ptr<SeriesStorage>(new ScalarSeriesStorage(*this));
    }
    std::unique_ptr<SeriesStorage> slice(std::size_t start, std::size_t end) const {
        if (end > data_.size()) end = data_.size();
        ScalarSeriesStorage* s = new ScalarSeriesStorage();
        s->data_.assign(data_.begin() + start, data_.begin() + end);
        return std::unique_ptr<SeriesStorage>(s);
    }

    T& at(std::size_t i) {
        if (i >= data_.size()) throw std::out_of_range("scalar index out of range");
        return data_[i];
    }
    const T& at(std::size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("scalar index out of range");
        return data_[i];
    }

    void append(const T& val) { data_.push_back(val); }
    void fill(const T& val) { std::fill(data_.begin(), data_.end(), val); }

    const std::vector<T>& values() const { return data_; }

    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

private:
    std::vector<T> data_;
};

// ==========================================================================
// VectorSeriesStorage<T>   numeric: stores std::vector<Eigen::VectorX<T>>
// ==========================================================================

template <typename T>
class VectorSeriesStorage : public SeriesStorage {
    static_assert(IsNumeric<T>::value, "primary VectorSeriesStorage is for numeric types only");

public:
    typedef Eigen::VectorX<T> RowType;

    explicit VectorSeriesStorage(Index width) : width_(width) {}
    VectorSeriesStorage(std::size_t rows, Index width, const T& fill_val = T()) : width_(width) {
        data_.reserve(rows);
        for (std::size_t i = 0; i < rows; ++i)
            data_.push_back(RowType::Constant(width, fill_val));
    }

    CellKind kind() const { return CellKind::kVector; }
    const std::type_info& value_type() const { return typeid(T); }
    std::string dtype_name() const { return typeid(T).name(); }
    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    std::vector<Index> cell_shape() const { return std::vector<Index>(1, width_); }
    Index values_per_row() const { return width_; }
    Index width() const { return width_; }

    void resize(std::size_t n) {
        if (n > data_.size())
            data_.resize(n, RowType::Zero(width_));
        else
            data_.resize(n);
    }
    void clear() { data_.clear(); }

    void append_from_row(const SeriesStorage& src, std::size_t src_index) {
        const VectorSeriesStorage<T>* p = dynamic_cast<const VectorSeriesStorage<T>*>(&src);
        if (!p) throw std::bad_cast();
        append(p->at(src_index));
    }

    std::unique_ptr<SeriesStorage> clone() const {
        return std::unique_ptr<SeriesStorage>(new VectorSeriesStorage(*this));
    }
    std::unique_ptr<SeriesStorage> slice(std::size_t start, std::size_t end) const {
        if (end > data_.size()) end = data_.size();
        VectorSeriesStorage* s = new VectorSeriesStorage(width_);
        s->data_.assign(data_.begin() + start, data_.begin() + end);
        return std::unique_ptr<SeriesStorage>(s);
    }

    RowType& at(std::size_t i) {
        if (i >= data_.size()) throw std::out_of_range("vector row index out of range");
        return data_[i];
    }
    const RowType& at(std::size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("vector row index out of range");
        return data_[i];
    }

    void append(const RowType& v) {
        if (v.size() != width_) throw std::invalid_argument("vector width mismatch");
        data_.push_back(v);
    }
    void fill(const T& val) {
        for (std::size_t i = 0; i < data_.size(); ++i)
            data_[i].setConstant(val);
    }

    typedef typename std::vector<RowType>::iterator iterator;
    typedef typename std::vector<RowType>::const_iterator const_iterator;
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

private:
    std::vector<RowType> data_;
    Index width_;
};

// ==========================================================================
// VectorSeriesStorage<std::string>  stores std::vector<std::vector<std::string>>
// ==========================================================================

template <>
class VectorSeriesStorage<std::string> : public SeriesStorage {
public:
    typedef std::vector<std::string> RowType;

    explicit VectorSeriesStorage(Index width) : width_(width) {}
    VectorSeriesStorage(std::size_t rows, Index width,
                     const std::string& fill_val = std::string())
        : data_(rows, RowType(static_cast<std::size_t>(width), fill_val)), width_(width) {}

    CellKind kind() const { return CellKind::kVector; }
    const std::type_info& value_type() const { return typeid(std::string); }
    std::string dtype_name() const { return "string"; }
    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    std::vector<Index> cell_shape() const { return std::vector<Index>(1, width_); }
    Index values_per_row() const { return width_; }
    Index width() const { return width_; }

    void resize(std::size_t n) { data_.resize(n, RowType(static_cast<std::size_t>(width_))); }
    void clear() { data_.clear(); }

    void append_from_row(const SeriesStorage& src, std::size_t src_index) {
        const VectorSeriesStorage<std::string>* p =
            dynamic_cast<const VectorSeriesStorage<std::string>*>(&src);
        if (!p) throw std::bad_cast();
        append(p->at(src_index));
    }

    std::unique_ptr<SeriesStorage> clone() const {
        return std::unique_ptr<SeriesStorage>(new VectorSeriesStorage(*this));
    }
    std::unique_ptr<SeriesStorage> slice(std::size_t start, std::size_t end) const {
        if (end > data_.size()) end = data_.size();
        VectorSeriesStorage* s = new VectorSeriesStorage(width_);
        s->data_.assign(data_.begin() + start, data_.begin() + end);
        return std::unique_ptr<SeriesStorage>(s);
    }

    RowType& at(std::size_t i) {
        if (i >= data_.size()) throw std::out_of_range("vector row index out of range");
        return data_[i];
    }
    const RowType& at(std::size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("vector row index out of range");
        return data_[i];
    }

    void append(const RowType& v) {
        if (static_cast<Index>(v.size()) != width_)
            throw std::invalid_argument("vector width mismatch");
        data_.push_back(v);
    }
    void fill(const std::string& val) {
        for (std::size_t i = 0; i < data_.size(); ++i)
            std::fill(data_[i].begin(), data_[i].end(), val);
    }

    typedef std::vector<RowType>::iterator iterator;
    typedef std::vector<RowType>::const_iterator const_iterator;
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

private:
    std::vector<RowType> data_;
    Index width_;
};

// ==========================================================================
// MatrixSeriesStorage<T>   numeric: stores std::vector<Eigen::MatrixX<T>>
// ==========================================================================

template <typename T>
class MatrixSeriesStorage : public SeriesStorage {
    static_assert(IsNumeric<T>::value, "primary MatrixSeriesStorage is for numeric types only");

public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> RowType;

    MatrixSeriesStorage(Index cell_rows, Index cell_cols)
        : cell_rows_(cell_rows), cell_cols_(cell_cols) {}
    MatrixSeriesStorage(std::size_t rows, Index cell_rows, Index cell_cols,
                     const T& fill_val = T())
        : cell_rows_(cell_rows), cell_cols_(cell_cols) {
        data_.reserve(rows);
        for (std::size_t i = 0; i < rows; ++i)
            data_.push_back(RowType::Constant(cell_rows, cell_cols, fill_val));
    }

    CellKind kind() const { return CellKind::kMatrix; }
    const std::type_info& value_type() const { return typeid(T); }
    std::string dtype_name() const { return typeid(T).name(); }
    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    std::vector<Index> cell_shape() const {
        std::vector<Index> s;
        s.push_back(cell_rows_);
        s.push_back(cell_cols_);
        return s;
    }
    Index values_per_row() const { return cell_rows_ * cell_cols_; }
    Index cell_rows() const { return cell_rows_; }
    Index cell_cols() const { return cell_cols_; }

    void resize(std::size_t n) {
        if (n > data_.size())
            data_.resize(n, RowType::Zero(cell_rows_, cell_cols_));
        else
            data_.resize(n);
    }
    void clear() { data_.clear(); }

    void append_from_row(const SeriesStorage& src, std::size_t src_index) {
        const MatrixSeriesStorage<T>* p = dynamic_cast<const MatrixSeriesStorage<T>*>(&src);
        if (!p) throw std::bad_cast();
        append(p->at(src_index));
    }

    std::unique_ptr<SeriesStorage> clone() const {
        return std::unique_ptr<SeriesStorage>(new MatrixSeriesStorage(*this));
    }
    std::unique_ptr<SeriesStorage> slice(std::size_t start, std::size_t end) const {
        if (end > data_.size()) end = data_.size();
        MatrixSeriesStorage* s = new MatrixSeriesStorage(cell_rows_, cell_cols_);
        s->data_.assign(data_.begin() + start, data_.begin() + end);
        return std::unique_ptr<SeriesStorage>(s);
    }

    RowType& at(std::size_t i) {
        if (i >= data_.size()) throw std::out_of_range("matrix row index out of range");
        return data_[i];
    }
    const RowType& at(std::size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("matrix row index out of range");
        return data_[i];
    }

    void append(const RowType& m) {
        if (m.rows() != cell_rows_ || m.cols() != cell_cols_)
            throw std::invalid_argument("matrix shape mismatch");
        data_.push_back(m);
    }
    void fill(const T& val) {
        for (std::size_t i = 0; i < data_.size(); ++i)
            data_[i].setConstant(val);
    }

    typedef typename std::vector<RowType>::iterator iterator;
    typedef typename std::vector<RowType>::const_iterator const_iterator;
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

private:
    std::vector<RowType> data_;
    Index cell_rows_, cell_cols_;
};

// ==========================================================================
// MatrixSeriesStorage<std::string>  stores std::vector<std::vector<std::string>> (row-major flat)
// ==========================================================================

template <>
class MatrixSeriesStorage<std::string> : public SeriesStorage {
public:
    typedef std::vector<std::string> RowType;  // row-major flat

    MatrixSeriesStorage(Index cell_rows, Index cell_cols)
        : cell_rows_(cell_rows), cell_cols_(cell_cols) {}
    MatrixSeriesStorage(std::size_t rows, Index cell_rows, Index cell_cols,
                     const std::string& fill_val = std::string())
        : data_(rows, RowType(static_cast<std::size_t>(cell_rows * cell_cols), fill_val)), cell_rows_(cell_rows),
          cell_cols_(cell_cols) {}

    CellKind kind() const { return CellKind::kMatrix; }
    const std::type_info& value_type() const { return typeid(std::string); }
    std::string dtype_name() const { return "string"; }
    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    std::vector<Index> cell_shape() const {
        std::vector<Index> s;
        s.push_back(cell_rows_);
        s.push_back(cell_cols_);
        return s;
    }
    Index values_per_row() const { return cell_rows_ * cell_cols_; }
    Index cell_rows() const { return cell_rows_; }
    Index cell_cols() const { return cell_cols_; }

    void resize(std::size_t n) {
        data_.resize(n, RowType(static_cast<std::size_t>(cell_rows_ * cell_cols_)));
    }
    void clear() { data_.clear(); }

    void append_from_row(const SeriesStorage& src, std::size_t src_index) {
        const MatrixSeriesStorage<std::string>* p =
            dynamic_cast<const MatrixSeriesStorage<std::string>*>(&src);
        if (!p) throw std::bad_cast();
        append(p->at(src_index));
    }

    std::unique_ptr<SeriesStorage> clone() const {
        return std::unique_ptr<SeriesStorage>(new MatrixSeriesStorage(*this));
    }
    std::unique_ptr<SeriesStorage> slice(std::size_t start, std::size_t end) const {
        if (end > data_.size()) end = data_.size();
        MatrixSeriesStorage* s = new MatrixSeriesStorage(cell_rows_, cell_cols_);
        s->data_.assign(data_.begin() + start, data_.begin() + end);
        return std::unique_ptr<SeriesStorage>(s);
    }

    // Row access: at(i) -> flat vector<string>
    RowType& at(std::size_t i) {
        if (i >= data_.size()) throw std::out_of_range("matrix row index out of range");
        return data_[i];
    }
    const RowType& at(std::size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("matrix row index out of range");
        return data_[i];
    }

    // Element access by [i][r][c]
    std::string& at(std::size_t i, Index r, Index c) {
        return at(i)[static_cast<std::size_t>(r * cell_cols_ + c)];
    }
    const std::string& at(std::size_t i, Index r, Index c) const {
        return at(i)[static_cast<std::size_t>(r * cell_cols_ + c)];
    }

    void append(const RowType& row) {
        if (static_cast<Index>(row.size()) != cell_rows_ * cell_cols_)
            throw std::invalid_argument("matrix flat size mismatch");
        data_.push_back(row);
    }
    void fill(const std::string& val) {
        for (std::size_t i = 0; i < data_.size(); ++i)
            std::fill(data_[i].begin(), data_[i].end(), val);
    }

    typedef std::vector<RowType>::iterator iterator;
    typedef std::vector<RowType>::const_iterator const_iterator;
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

private:
    std::vector<RowType> data_;
    Index cell_rows_, cell_cols_;
};

// ==========================================================================
// CellSeries  -- type-erased facade, holds no template parameters
// ==========================================================================

class CellSeries {
public:
    class RowRef {
    public:
        friend class CellSeries;
        RowRef(CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}

        CellKind kind() const { return owner_->cell_kind(); }
        const std::type_info& dtype() const { return owner_->dtype(); }
        std::size_t index() const { return idx_; }

        template <typename T>
        T& scalar() {
            check_access<T>(CellKind::kScalar);
            return owner_->template scalar_at<T>(idx_);
        }

        template <typename T>
        typename VectorRowType<T>::type& vector() {
            check_access<T>(CellKind::kVector);
            return owner_->template vec_at<T>(idx_);
        }

        template <typename T>
        typename MatrixRowType<T>::type& matrix() {
            check_access<T>(CellKind::kMatrix);
            return owner_->template mat_at<T>(idx_);
        }

    private:
        template <typename T>
        void check_access(CellKind expected) const {
            if (owner_->cell_kind() != expected || owner_->dtype() != typeid(T)) throw std::bad_cast();
        }

        CellSeries* owner_;
        std::size_t idx_;
    };

    class ConstRowRef {
    public:
        ConstRowRef(const CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}

        CellKind kind() const { return owner_->cell_kind(); }
        const std::type_info& dtype() const { return owner_->dtype(); }
        std::size_t index() const { return idx_; }

        template <typename T>
        const T& scalar() const {
            check_access<T>(CellKind::kScalar);
            return owner_->template scalar_at<T>(idx_);
        }

        template <typename T>
        const typename VectorRowType<T>::type& vector() const {
            check_access<T>(CellKind::kVector);
            return owner_->template vec_at<T>(idx_);
        }

        template <typename T>
        const typename MatrixRowType<T>::type& matrix() const {
            check_access<T>(CellKind::kMatrix);
            return owner_->template mat_at<T>(idx_);
        }

    private:
        template <typename T>
        void check_access(CellKind expected) const {
            if (owner_->cell_kind() != expected || owner_->dtype() != typeid(T)) throw std::bad_cast();
        }

        const CellSeries* owner_;
        std::size_t idx_;
    };

    class iterator {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef RowRef value_type;
        typedef std::ptrdiff_t difference_type;
        typedef void pointer;
        typedef RowRef reference;

        iterator() : owner_(nullptr), idx_(0) {}
        iterator(CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}

        reference operator*() const { return RowRef(owner_, idx_); }

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

        CellSeries* owner() const { return owner_; }
        std::size_t index() const { return idx_; }

    private:
        CellSeries* owner_;
        std::size_t idx_;
    };

    class const_iterator {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef ConstRowRef value_type;
        typedef std::ptrdiff_t difference_type;
        typedef void pointer;
        typedef ConstRowRef reference;

        const_iterator() : owner_(nullptr), idx_(0) {}
        const_iterator(const CellSeries* owner, std::size_t idx) : owner_(owner), idx_(idx) {}
        const_iterator(const iterator& it) : owner_(it.owner()), idx_(it.index()) {}

        reference operator*() const { return ConstRowRef(owner_, idx_); }

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

    CellSeries() {}

    CellSeries(const CellSeries& other)
        : storage_(other.storage_ ? other.storage_->clone() : std::unique_ptr<SeriesStorage>()) {}

    CellSeries& operator=(const CellSeries& other) {
        if (this != &other)
            storage_ = other.storage_ ? other.storage_->clone() : std::unique_ptr<SeriesStorage>();
        return *this;
    }

    CellSeries(CellSeries&& other) noexcept : storage_(std::move(other.storage_)) {}

    CellSeries& operator=(CellSeries&& other) noexcept {
        storage_ = std::move(other.storage_);
        return *this;
    }

    // ---- Factory functions ----

    // Scalar series: each row holds a single value of type T
    template <typename T>
    static CellSeries Scalars(std::size_t rows, const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s;
        s.storage_.reset(new ScalarSeriesStorage<T>(rows, fill_val));
        return s;
    }

    // Construct a scalar series from an existing vector
    template <typename T>
    static CellSeries From(const std::vector<T>& values) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s;
        s.storage_.reset(new ScalarSeriesStorage<T>(values));
        return s;
    }

    // Vector series: each row holds a vector of length width
    template <typename T>
    static CellSeries Vectors(std::size_t rows, Index width, const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s;
        s.storage_.reset(new VectorSeriesStorage<T>(rows, width, fill_val));
        return s;
    }

    // Matrix series: each row holds a cell_rows x cell_cols matrix
    template <typename T>
    static CellSeries Matrices(std::size_t rows, Index cell_rows, Index cell_cols,
                                  const T& fill_val = T()) {
        static_assert(IsSupported<T>::value, "unsupported type");
        CellSeries s;
        s.storage_.reset(new MatrixSeriesStorage<T>(rows, cell_rows, cell_cols, fill_val));
        return s;
    }

    // ---- Properties (pandas-like) ----

    bool has_value() const { return storage_.get() != nullptr; }
    std::size_t size() const { return storage_ref().size(); }
    bool empty() const { return !storage_ || storage_->empty(); }
    CellKind cell_kind() const { return storage_ref().kind(); }

    // Shape of each cell: scalar->[], vector->[n], matrix->[r,c]
    std::vector<Index> cell_shape() const { return storage_ref().cell_shape(); }
    Index values_per_row() const { return storage_ref().values_per_row(); }
    std::string dtype_name() const { return storage_ref().dtype_name(); }
    const std::type_info& dtype() const { return storage_ref().value_type(); }

    // ---- Resize / clear ----

    void resize(std::size_t n) { storage_mut().resize(n); }
    void clear() { storage_mut().clear(); }

    // ---- Slicing (pandas iloc) ----

    CellSeries head(std::size_t n) const {
        CellSeries s;
        s.storage_ = storage_ref().slice(0, n);
        return s;
    }

    CellSeries tail(std::size_t n) const {
        std::size_t sz = size();
        CellSeries s;
        s.storage_ = storage_ref().slice(sz > n ? sz - n : 0, sz);
        return s;
    }

    CellSeries iloc(std::size_t start, std::size_t end) const {
        CellSeries s;
        s.storage_ = storage_ref().slice(start, end);
        return s;
    }

    iterator begin() {
        if (!has_value()) return iterator(this, 0);
        return iterator(this, 0);
    }
    iterator end() {
        if (!has_value()) return iterator(this, 0);
        return iterator(this, size());
    }
    const_iterator begin() const {
        if (!has_value()) return const_iterator(this, 0);
        return const_iterator(this, 0);
    }
    const_iterator end() const {
        if (!has_value()) return const_iterator(this, 0);
        return const_iterator(this, size());
    }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // ---- Scalar row access ----

    template <typename T>
    T& scalar_at(std::size_t i) { return scalar_storage<T>().at(i); }
    template <typename T>
    const T& scalar_at(std::size_t i) const { return scalar_storage<T>().at(i); }

    // ---- Vector row access ----
    // numeric T -> Eigen::VectorX<T>&
    // string   -> std::vector<std::string>&

    template <typename T>
    typename VectorRowType<T>::type& vec_at(std::size_t i) {
        return vector_storage<T>().at(i);
    }
    template <typename T>
    const typename VectorRowType<T>::type& vec_at(std::size_t i) const {
        return vector_storage<T>().at(i);
    }

    // ---- Matrix row access ----
    // numeric T -> Eigen::Matrix<T,Dynamic,Dynamic>&
    // string   -> std::vector<std::string>& (row-major flat)

    template <typename T>
    typename MatrixRowType<T>::type& mat_at(std::size_t i) {
        return matrix_storage<T>().at(i);
    }
    template <typename T>
    const typename MatrixRowType<T>::type& mat_at(std::size_t i) const {
        return matrix_storage<T>().at(i);
    }

    // ---- Append ----

    template <typename T>
    typename std::enable_if<IsSupported<T>::value, void>::type
    append(const T& val) { scalar_storage<T>().append(val); }

    // Single-argument append: directly append a numeric vector row
    template <typename T>
    void append(const Eigen::VectorX<T>& v) {
        vector_storage<T>().append(v);
    }

    // Single-argument append: directly append a numeric matrix row
    template <typename T>
    void append(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
        matrix_storage<T>().append(m);
    }

    // Single-argument append: dispatch string row to vector or matrix string storage based on cell_kind
    void append(const std::vector<std::string>& row) {
        switch (cell_kind()) {
            case CellKind::kVector: vector_storage<std::string>().append(row); break;
            case CellKind::kMatrix: matrix_storage<std::string>().append(row); break;
            default: throw std::bad_cast();
        }
    }

    template <typename T>
    void append_vec(const typename VectorRowType<T>::type& v) {
        vector_storage<T>().append(v);
    }

    template <typename T>
    void append_mat(const typename MatrixRowType<T>::type& m) {
        matrix_storage<T>().append(m);
    }

    // ---- Fill all elements ----

    template <typename T>
    void fill(const T& val) {
        switch (storage_ref().kind()) {
            case CellKind::kScalar: scalar_storage<T>().fill(val); break;
            case CellKind::kVector: vector_storage<T>().fill(val); break;
            case CellKind::kMatrix: matrix_storage<T>().fill(val); break;
        }
    }

    // ---- Access typed storage (advanced use) ----

    template <typename T>
    ScalarSeriesStorage<T>& as_scalar_storage() { return scalar_storage<T>(); }
    template <typename T>
    const ScalarSeriesStorage<T>& as_scalar_storage() const { return scalar_storage<T>(); }

    template <typename T>
    VectorSeriesStorage<T>& as_vector_storage() { return vector_storage<T>(); }
    template <typename T>
    const VectorSeriesStorage<T>& as_vector_storage() const { return vector_storage<T>(); }

    template <typename T>
    MatrixSeriesStorage<T>& as_matrix_storage() { return matrix_storage<T>(); }
    template <typename T>
    const MatrixSeriesStorage<T>& as_matrix_storage() const { return matrix_storage<T>(); }

private:
    SeriesStorage& storage_mut() {
        if (!storage_) throw std::runtime_error("CellSeries has no value");
        return *storage_;
    }
    const SeriesStorage& storage_ref() const {
        if (!storage_) throw std::runtime_error("CellSeries has no value");
        return *storage_;
    }

    template <typename T>
    ScalarSeriesStorage<T>& scalar_storage() {
        ScalarSeriesStorage<T>* p = dynamic_cast<ScalarSeriesStorage<T>*>(storage_.get());
        if (!p) throw std::bad_cast();
        return *p;
    }
    template <typename T>
    const ScalarSeriesStorage<T>& scalar_storage() const {
        const ScalarSeriesStorage<T>* p = dynamic_cast<const ScalarSeriesStorage<T>*>(storage_.get());
        if (!p) throw std::bad_cast();
        return *p;
    }

    template <typename T>
    VectorSeriesStorage<T>& vector_storage() {
        VectorSeriesStorage<T>* p = dynamic_cast<VectorSeriesStorage<T>*>(storage_.get());
        if (!p) throw std::bad_cast();
        return *p;
    }
    template <typename T>
    const VectorSeriesStorage<T>& vector_storage() const {
        const VectorSeriesStorage<T>* p = dynamic_cast<const VectorSeriesStorage<T>*>(storage_.get());
        if (!p) throw std::bad_cast();
        return *p;
    }

    template <typename T>
    MatrixSeriesStorage<T>& matrix_storage() {
        MatrixSeriesStorage<T>* p = dynamic_cast<MatrixSeriesStorage<T>*>(storage_.get());
        if (!p) throw std::bad_cast();
        return *p;
    }
    template <typename T>
    const MatrixSeriesStorage<T>& matrix_storage() const {
        const MatrixSeriesStorage<T>* p = dynamic_cast<const MatrixSeriesStorage<T>*>(storage_.get());
        if (!p) throw std::bad_cast();
        return *p;
    }

    std::unique_ptr<SeriesStorage> storage_;
};

}  // namespace xdataset

#endif  // XDATASET_TENSOR_SERIES_H_
