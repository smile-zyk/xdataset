#include "data_series.h"

#include <algorithm>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xdataset {

// =========================================================================
// DataSeries — constructors & assignment operators
// =========================================================================

DataSeries::DataSeries()
    : kind_(DataKind::kScalar),
      dtype_(DTypeTag::kReal),
      shape_(),
      storage_(make_storage(DataKind::kScalar, DTypeTag::kReal, std::vector<Index>())),
      unit_() {}

DataSeries::DataSeries(DataKind kind, DTypeTag dtype, const std::vector<Index>& shape)
    : kind_(kind), dtype_(dtype), shape_(shape), storage_(make_storage(kind, dtype, shape)), unit_() {}

DataSeries::DataSeries(const DataSeries& other)
    : kind_(other.kind_), dtype_(other.dtype_), shape_(other.shape_),
      storage_(other.storage_->clone()), unit_(other.unit_) {}

DataSeries& DataSeries::operator=(const DataSeries& other) {
    if (this != &other) {
        kind_ = other.kind_;
        dtype_ = other.dtype_;
        shape_ = other.shape_;
        storage_ = other.storage_->clone();
        unit_ = other.unit_;
    }
    return *this;
}

DataSeries::DataSeries(DataSeries&& other) noexcept
    : kind_(other.kind_), dtype_(other.dtype_),
      shape_(std::move(other.shape_)),
      storage_(std::move(other.storage_)),
      unit_(std::move(other.unit_)) {}

DataSeries& DataSeries::operator=(DataSeries&& other) noexcept {
    kind_ = other.kind_;
    dtype_ = other.dtype_;
    shape_ = std::move(other.shape_);
    storage_ = std::move(other.storage_);
    unit_ = std::move(other.unit_);
    return *this;
}

// =========================================================================
// DataSeries — unit
// =========================================================================

void DataSeries::set_unit(const std::string& s) {
    unit_ = parse_unit(s);
}

void DataSeries::canonicalize() {
    // Strings are not numeric — only update the unit tag, no value conversion.
    if (dtype_ == DTypeTag::kString) {
        unit_ = xdataset::canonicalize(unit_);
        return;
    }

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

DataSeries DataSeries::canonicalized() const {
    DataSeries result(*this);
    result.canonicalize();
    return result;
}

bool DataSeries::is_canonicalized() const {
    return is_canonical(unit_);
}

// =========================================================================
// DataSeries — slicing
// =========================================================================

DataSeries DataSeries::head(std::size_t n) const {
    std::size_t sz = size();
    return iloc(0, n < sz ? n : sz);
}

DataSeries DataSeries::tail(std::size_t n) const {
    std::size_t sz = size();
    return iloc(n < sz ? sz - n : 0, sz);
}

DataSeries DataSeries::iloc(std::size_t start, std::size_t end) const {
    if (start > end || end > size()) throw std::out_of_range("iloc out of range");
    DataSeries out(kind_, dtype_, shape_);
    out.unit_ = unit_;
    for (std::size_t i = start; i < end; ++i) out.append_from(*this, static_cast<Index>(i));
    return out;
}

// =========================================================================
// DataSeries — append helpers (non-template overloads)
// =========================================================================

void DataSeries::append_vector(const Eigen::Tensor<std::string, 1>& v) {
    if (v.dimension(0) != element_count()) throw std::bad_cast();
    vector_storage_string()->append(v);
}

void DataSeries::append_matrix(const Eigen::Tensor<std::string, 2>& m) {
    if (m.dimension(0) != shape_[0] || m.dimension(1) != shape_[1]) throw std::bad_cast();
    matrix_storage_string()->append(m);
}

void DataSeries::append_from(const DataSeries& src, Index row) {
    const Index dst_row = static_cast<Index>(size());
    resize(size() + 1);
    assign_from(src, row, dst_row);
}

void DataSeries::assign_from(const DataSeries& src, Index src_row, Index dst_row) {
    if (src.kind_ != kind_ || src.dtype_ != dtype_ || src.shape_ != shape_) throw std::bad_cast();
    if (src_row < 0 || static_cast<std::size_t>(src_row) >= src.size() ||
        dst_row < 0 || static_cast<std::size_t>(dst_row) >= size()) throw std::out_of_range("row index out of range");

    // First non-dimensionless source sets the series' unit.
    if (is_dimensionless(unit_) && !is_dimensionless(src.unit_)) {
        if (dtype_ == DTypeTag::kString)
            throw std::invalid_argument("string series cannot have a named unit");
        unit_ = src.unit_;
    } else if (!same_dimension(src.unit_, unit_)) {
        throw std::invalid_argument(
            "unit mismatch: series has dimension [" + unit_string(unit_) +
            "], source has [" + unit_string(src.unit_) + "]");
    }

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

void DataSeries::append(const Measurement& m) {
    if (m.kind() != kind_ || m.dtype() != dtype_ || m.shape() != shape_) throw std::bad_cast();

    // First non-dimensionless measurement sets the series' unit.
    if (is_dimensionless(unit_) && !is_dimensionless(m.unit())) {
        if (dtype_ == DTypeTag::kString)
            throw std::invalid_argument("string series cannot have a named unit");
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

// =========================================================================
// DataSeries — contiguous data
// =========================================================================

std::size_t DataSeries::contiguous_bytes() const {
    if (dtype_ == DTypeTag::kReal) return contiguous_elements() * sizeof(double);
    if (dtype_ == DTypeTag::kInteger) return contiguous_elements() * sizeof(int);
    if (dtype_ == DTypeTag::kComplex) return contiguous_elements() * sizeof(std::complex<double>);
    throw std::runtime_error("string storage is not trivially-copyable");
}

// =========================================================================
// DataSeries — measurement_at
// =========================================================================

Measurement DataSeries::measurement_at(Index i) const {
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

// =========================================================================
// DataSeries — at (public)
// =========================================================================

DataSeries DataSeries::at(const std::vector<Index>& selected, bool reduce_to_scalar) const {
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

DataSeries DataSeries::at(
    const std::vector<Index>& selected_rows,
    const std::vector<Index>& selected_cols,
    bool row_reduce,
    bool col_reduce) const {
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

// =========================================================================
// DataSeries — at (private helpers)
// =========================================================================

DataSeries DataSeries::at_vector_string_impl(const std::vector<Index>& selected) const {
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

DataSeries DataSeries::at_matrix_string_impl(
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

// =========================================================================
// DataSeries — private: validate_schema / make_storage
// =========================================================================

void DataSeries::validate_schema(DataKind kind, const std::vector<Index>& shape) {
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

std::unique_ptr<SeriesStorage> DataSeries::make_storage(DataKind kind, DTypeTag dtype, const std::vector<Index>& shape) {
    validate_schema(kind, shape);
    if (kind == DataKind::kScalar) {
        if (dtype == DTypeTag::kReal) return std::unique_ptr<SeriesStorage>(new ScalarSeriesStorage<double>());
        if (dtype == DTypeTag::kInteger) return std::unique_ptr<SeriesStorage>(new ScalarSeriesStorage<int>());
        if (dtype == DTypeTag::kComplex) return std::unique_ptr<SeriesStorage>(new ScalarSeriesStorage<std::complex<double> >());
        return std::unique_ptr<SeriesStorage>(new ScalarSeriesStorage<std::string>());
    }

    if (kind == DataKind::kVector) {
        if (dtype == DTypeTag::kReal) return std::unique_ptr<SeriesStorage>(new VectorNumericSeriesStorage<double>(shape[0]));
        if (dtype == DTypeTag::kInteger) return std::unique_ptr<SeriesStorage>(new VectorNumericSeriesStorage<int>(shape[0]));
        if (dtype == DTypeTag::kComplex) return std::unique_ptr<SeriesStorage>(new VectorNumericSeriesStorage<std::complex<double> >(shape[0]));
        return std::unique_ptr<SeriesStorage>(new VectorStringSeriesStorage(shape[0]));
    }

    if (dtype == DTypeTag::kReal) return std::unique_ptr<SeriesStorage>(new MatrixNumericSeriesStorage<double>(shape[0], shape[1]));
    if (dtype == DTypeTag::kInteger) return std::unique_ptr<SeriesStorage>(new MatrixNumericSeriesStorage<int>(shape[0], shape[1]));
    if (dtype == DTypeTag::kComplex) return std::unique_ptr<SeriesStorage>(new MatrixNumericSeriesStorage<std::complex<double> >(shape[0], shape[1]));
    return std::unique_ptr<SeriesStorage>(new MatrixStringSeriesStorage(shape[0], shape[1]));
}

// =========================================================================
// DataSeries — private: string storage accessors
// =========================================================================

VectorStringSeriesStorage* DataSeries::vector_storage_string() {
    if (kind_ != DataKind::kVector || dtype_ != DTypeTag::kString) throw std::bad_cast();
    return static_cast<VectorStringSeriesStorage*>(storage_.get());
}

const VectorStringSeriesStorage* DataSeries::vector_storage_string() const {
    if (kind_ != DataKind::kVector || dtype_ != DTypeTag::kString) throw std::bad_cast();
    return static_cast<const VectorStringSeriesStorage*>(storage_.get());
}

MatrixStringSeriesStorage* DataSeries::matrix_storage_string() {
    if (kind_ != DataKind::kMatrix || dtype_ != DTypeTag::kString) throw std::bad_cast();
    return static_cast<MatrixStringSeriesStorage*>(storage_.get());
}

const MatrixStringSeriesStorage* DataSeries::matrix_storage_string() const {
    if (kind_ != DataKind::kMatrix || dtype_ != DTypeTag::kString) throw std::bad_cast();
    return static_cast<const MatrixStringSeriesStorage*>(storage_.get());
}

// =========================================================================
// DataSeries — private: fill helpers (non-template overloads)
// =========================================================================

void DataSeries::fill_vector_row(Index row, const std::string& val, std::true_type) {
    Eigen::Tensor<std::string, 1>& t = vector_at<std::string>(row);
    for (Index i = 0; i < t.dimension(0); ++i) t(i) = val;
}

void DataSeries::fill_matrix_row(Index row, const std::string& val, std::true_type) {
    Eigen::Tensor<std::string, 2>& t = matrix_at<std::string>(row);
    for (Index r = 0; r < t.dimension(0); ++r) {
        for (Index c = 0; c < t.dimension(1); ++c) {
            t(r, c) = val;
        }
    }
}

// =========================================================================
// DataSeries — static factories (non-template overloads)
// =========================================================================

DataSeries DataSeries::CreateScalarFromVector(const std::vector<std::string>& values,
                                             const Unit& u) {
    DataSeries s(DataKind::kScalar, DTypeTag::kString, {});
    s.set_unit(u);
    s.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        s.scalar_at<std::string>(static_cast<Index>(i)) = values[i];
    return s;
}

DataSeries DataSeries::CreateVectorFromVector(Index width, const std::vector<std::string>& values,
                                             const Unit& u) {
    if (width < 0)
        throw std::invalid_argument("vector width must be non-negative");
    const std::size_t w = static_cast<std::size_t>(width);
    if (w == 0) {
        if (!values.empty())
            throw std::invalid_argument("vector width 0 requires empty data");
        DataSeries s(DataKind::kVector, DTypeTag::kString, {width});
        s.set_unit(u);
        return s;
    }
    if (values.size() % w != 0)
        throw std::invalid_argument("vector flat data length must be a multiple of width");
    DataSeries s(DataKind::kVector, DTypeTag::kString, {width});
    s.set_unit(u);
    s.resize(values.size() / w);
    for (std::size_t i = 0; i < s.size(); ++i)
        for (Index j = 0; j < width; ++j)
            s.vector_at<std::string>(static_cast<Index>(i))(j) = values[i * w + j];
    return s;
}

DataSeries DataSeries::CreateMatrixFromVector(Index cell_rows, Index cell_cols,
                                             const std::vector<std::string>& values,
                                             const Unit& u) {
    if (cell_rows < 0 || cell_cols < 0)
        throw std::invalid_argument("matrix shape must be non-negative");
    const std::size_t elems = static_cast<std::size_t>(cell_rows) *
                              static_cast<std::size_t>(cell_cols);
    if (elems == 0) {
        if (!values.empty())
            throw std::invalid_argument("zero-sized matrix cells require empty data");
        DataSeries s(DataKind::kMatrix, DTypeTag::kString, {cell_rows, cell_cols});
        s.set_unit(u);
        return s;
    }
    if (values.size() % elems != 0)
        throw std::invalid_argument("matrix flat data length must be a multiple of cell_rows * cell_cols");
    DataSeries s(DataKind::kMatrix, DTypeTag::kString, {cell_rows, cell_cols});
    s.set_unit(u);
    s.resize(values.size() / elems);
    for (std::size_t i = 0; i < s.size(); ++i)
        for (Index r = 0; r < cell_rows; ++r)
            for (Index c = 0; c < cell_cols; ++c)
                s.matrix_at<std::string>(static_cast<Index>(i))(r, c) =
                    values[i * elems + static_cast<std::size_t>(r) *
                     static_cast<std::size_t>(cell_cols) + static_cast<std::size_t>(c)];
    return s;
}

DataSeries DataSeries::CreateVectorFromNestedVector(const std::vector<std::vector<std::string>>& rows,
                                                   const Unit& u) {
    if (rows.empty()) {
        DataSeries s(DataKind::kVector, DTypeTag::kString, {0});
        s.set_unit(u);
        return s;
    }
    const Index width = static_cast<Index>(rows[0].size());
    DataSeries s(DataKind::kVector, DTypeTag::kString, {width});
    s.set_unit(u);
    s.resize(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (static_cast<Index>(rows[i].size()) != width)
            throw std::invalid_argument("all vector rows must have the same width");
        for (Index j = 0; j < width; ++j)
            s.vector_at<std::string>(static_cast<Index>(i))(j) = rows[i][static_cast<std::size_t>(j)];
    }
    return s;
}

DataSeries DataSeries::CreateMatrixFromNestedVector(
    const std::vector<std::vector<std::vector<std::string>>>& rows,
    const Unit& u) {
    if (rows.empty()) {
        DataSeries s(DataKind::kMatrix, DTypeTag::kString, {0, 0});
        s.set_unit(u);
        return s;
    }
    const Index cell_rows = static_cast<Index>(rows[0].size());
    const Index cell_cols = cell_rows > 0 ? static_cast<Index>(rows[0][0].size()) : 0;
    DataSeries s(DataKind::kMatrix, DTypeTag::kString, {cell_rows, cell_cols});
    s.set_unit(u);
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

}  // namespace xdataset
