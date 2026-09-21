#include "measurement.h"
#include "unit.h"
#include "data_frame.h"

#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace xdataset
{

    // =========================================================================
    // MeasurementTypeVisitor -- extracts DataKind / DataType from a variant.
    // (internal: defined here, not in the public header)
    // =========================================================================

    struct MeasurementTypeVisitor : public boost::static_visitor<void>
    {
        DataKind kind   = DataKind::kScalar;
        DataType dtype  = DataType::kReal;

        void operator()(double)                    { kind = DataKind::kScalar;  dtype = DataType::kReal;    }
        void operator()(int)                       { kind = DataKind::kScalar;  dtype = DataType::kInteger; }
        void operator()(const std::complex<double>&){ kind = DataKind::kScalar;  dtype = DataType::kComplex; }
        void operator()(const std::string&)         { kind = DataKind::kScalar;  dtype = DataType::kString;  }
        void operator()(bool)                       { kind = DataKind::kScalar;  dtype = DataType::kBoolean; }

        void operator()(const VecXd&)           { kind = DataKind::kVector; dtype = DataType::kReal;    }
        void operator()(const VecXi&)           { kind = DataKind::kVector; dtype = DataType::kInteger; }
        void operator()(const VecXcd&)          { kind = DataKind::kVector; dtype = DataType::kComplex; }
        void operator()(const VecXs&)           { kind = DataKind::kVector; dtype = DataType::kString;  }

        void operator()(const MatXd&)           { kind = DataKind::kMatrix; dtype = DataType::kReal;    }
        void operator()(const MatXi&)           { kind = DataKind::kMatrix; dtype = DataType::kInteger; }
        void operator()(const MatXcd&)          { kind = DataKind::kMatrix; dtype = DataType::kComplex; }
        void operator()(const MatXs&)           { kind = DataKind::kMatrix; dtype = DataType::kString;  }
    };

    // =========================================================================
    // MeasurementEqualityVisitor -- element-wise comparison
    // =========================================================================
    //
    // Eigen matrices and Eigen::Tensor<std::string> do not support operator==,
    // so each variant alternative is compared element by element.

    struct MeasurementEqualityVisitor : public boost::static_visitor<bool>
    {
        const Measurement::Storage* rhs;

        explicit MeasurementEqualityVisitor(const Measurement::Storage* other)
            : rhs(other)
        {
        }

        // --- scalars (compiler-provided ==) ---
        bool operator()(double lhs) const
        {
            const double* rp = boost::get<double>(rhs);
            return rp != nullptr && lhs == *rp;
        }
        bool operator()(int lhs) const
        {
            const int* rp = boost::get<int>(rhs);
            return rp != nullptr && lhs == *rp;
        }
        bool operator()(const std::complex<double>& lhs) const
        {
            const std::complex<double>* rp = boost::get<std::complex<double>>(rhs);
            return rp != nullptr && lhs == *rp;
        }
        bool operator()(const std::string& lhs) const
        {
            const std::string* rp = boost::get<std::string>(rhs);
            return rp != nullptr && lhs == *rp;
        }
        bool operator()(bool lhs) const
        {
            const bool* rp = boost::get<bool>(rhs);
            return rp != nullptr && lhs == *rp;
        }

        // --- vectors ---
        bool operator()(const VecXd& lhs)   const { return compare_vector<double>(lhs); }
        bool operator()(const VecXi& lhs)   const { return compare_vector<int>(lhs); }
        bool operator()(const VecXcd& lhs)  const { return compare_vector<std::complex<double>>(lhs); }
        bool operator()(const VecXs& lhs)   const { return compare_string_vector(lhs); }

        // --- matrices ---
        bool operator()(const MatXd& lhs)   const { return compare_matrix<double>(lhs); }
        bool operator()(const MatXi& lhs)   const { return compare_matrix<int>(lhs); }
        bool operator()(const MatXcd& lhs)  const { return compare_matrix<std::complex<double>>(lhs); }
        bool operator()(const MatXs& lhs)   const { return compare_string_matrix(lhs); }

    private:
        // Helpers: the rhs alternative must be the same type; otherwise false.
        template <typename T>
        bool compare_vector(const Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>& lhs) const
        {
            const auto* rp = boost::get<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>>(rhs);
            if (!rp) return false;
            // Check size first: Eigen's operator== assumes matching sizes and
            // would assert/UB otherwise.
            if (lhs.size() != rp->size()) return false;
            // Eigen's == is a coefficient-wise expression; reduce with .all().
            return (lhs.array() == rp->array()).all();
        }

        template <typename T>
        bool compare_matrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& lhs) const
        {
            const auto* rp = boost::get<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rhs);
            if (!rp) return false;
            if (lhs.rows() != rp->rows() || lhs.cols() != rp->cols()) return false;
            return (lhs.array() == rp->array()).all();
        }

        bool compare_string_vector(const VecXs& lhs) const
        {
            const VecXs* rp = boost::get<VecXs>(rhs);
            if (!rp) return false;
            const VecXs& r = *rp;
            if (lhs.dimension(0) != r.dimension(0)) return false;
            for (Eigen::Index i = 0; i < lhs.dimension(0); ++i)
                if (lhs(i) != r(i)) return false;
            return true;
        }

        bool compare_string_matrix(const MatXs& lhs) const
        {
            const MatXs* rp = boost::get<MatXs>(rhs);
            if (!rp) return false;
            const MatXs& r = *rp;
            if (lhs.dimension(0) != r.dimension(0) || lhs.dimension(1) != r.dimension(1)) return false;
            for (Eigen::Index i = 0; i < lhs.dimension(0); ++i)
                for (Eigen::Index j = 0; j < lhs.dimension(1); ++j)
                    if (lhs(i, j) != r(i, j)) return false;
            return true;
        }
    };

    // =========================================================================
    // Measurement -- metadata inference
    // =========================================================================

    void Measurement::infer_metadata()
    {
        MeasurementTypeVisitor v;
        boost::apply_visitor(v, storage_);
        data_type_ = v.dtype;

        // Derive shape by dispatching on (kind, dtype).
        shape_.clear();
        DataKind kind = v.kind;
        switch (kind)
        {
            case DataKind::kScalar:
                break;  // shape_ remains empty

            case DataKind::kVector:
                switch (data_type_)
                {
                    case DataType::kReal:    shape_.push_back(boost::get<VecXd>(storage_).size());            break;
                    case DataType::kInteger: shape_.push_back(boost::get<VecXi>(storage_).size());            break;
                    case DataType::kComplex: shape_.push_back(boost::get<VecXcd>(storage_).size());           break;
                    case DataType::kString:  shape_.push_back(boost::get<VecXs>(storage_).dimension(0)); break;
                    default: break;  // kBoolean is scalar-only
                }
                break;

            case DataKind::kMatrix:
                switch (data_type_)
                {
                    case DataType::kReal:
                    {
                        const auto& m = boost::get<MatXd>(storage_);
                        shape_.push_back(m.rows()); shape_.push_back(m.cols());
                        break;
                    }
                    case DataType::kInteger:
                    {
                        const auto& m = boost::get<MatXi>(storage_);
                        shape_.push_back(m.rows()); shape_.push_back(m.cols());
                        break;
                    }
                    case DataType::kComplex:
                    {
                        const auto& m = boost::get<MatXcd>(storage_);
                        shape_.push_back(m.rows()); shape_.push_back(m.cols());
                        break;
                    }
                    case DataType::kString:
                    {
                        const auto& t = boost::get<MatXs>(storage_);
                        shape_.push_back(t.dimension(0)); shape_.push_back(t.dimension(1));
                        break;
                    }
                    default: break;  // kBoolean is scalar-only
                }
                break;
        }
    }

    // =========================================================================
    // Measurement -- default ctor
    // =========================================================================

    Measurement::Measurement()
        : data_type_(DataType::kReal)
        , shape_()
        , storage_(0.0)
        , unit_()
    {
    }

    // =========================================================================
    // Measurement -- static factories
    // =========================================================================

    Measurement Measurement::Real(double value, const Unit& u)
    {
        Measurement m;
        m.storage_ = value;
        m.data_type_   = DataType::kReal;
        m.unit_        = u;
        return m;
    }

    Measurement Measurement::Integer(int value, const Unit& u)
    {
        Measurement m;
        m.storage_ = value;
        m.data_type_   = DataType::kInteger;
        m.unit_        = u;
        return m;
    }

    Measurement Measurement::Complex(std::complex<double> value, const Unit& u)
    {
        Measurement m;
        m.storage_ = value;
        m.data_type_   = DataType::kComplex;
        m.unit_        = u;
        return m;
    }

    Measurement Measurement::String(std::string value)
    {
        Measurement m;
        m.storage_ = value;
        m.data_type_   = DataType::kString;
        return m;
    }

    Measurement Measurement::Boolean(bool value)
    {
        Measurement m;
        m.storage_ = value;
        m.data_type_   = DataType::kBoolean;
        return m;
    }

    Measurement Measurement::Vector(VecXd v, const Unit& u)
    {
        Measurement m;
        const Index sz = v.size();
        m.storage_ = std::move(v);
        m.data_type_   = DataType::kReal;
        m.unit_        = u;
        m.shape_.push_back(sz);
        return m;
    }

    Measurement Measurement::Vector(VecXi v, const Unit& u)
    {
        Measurement m;
        const Index sz = v.size();
        m.storage_ = std::move(v);
        m.data_type_   = DataType::kInteger;
        m.unit_        = u;
        m.shape_.push_back(sz);
        return m;
    }

    Measurement Measurement::Vector(VecXcd v, const Unit& u)
    {
        Measurement m;
        const Index sz = v.size();
        m.storage_ = std::move(v);
        m.data_type_   = DataType::kComplex;
        m.unit_        = u;
        m.shape_.push_back(sz);
        return m;
    }

    Measurement Measurement::Vector(VecConstMap<double> v, const Unit& u)
    {
        return Measurement::Vector(VecXd(v), u);
    }

    Measurement Measurement::Vector(VecConstMap<int> v, const Unit& u)
    {
        return Measurement::Vector(VecXi(v), u);
    }

    Measurement Measurement::Vector(VecConstMap<std::complex<double> > v,
                                    const Unit& u)
    {
        return Measurement::Vector(VecXcd(v), u);
    }

    Measurement Measurement::Vector(const VecXs& v)
    {
        Measurement m;
        m.storage_ = v;
        m.data_type_   = DataType::kString;
        m.shape_.push_back(v.dimension(0));
        return m;
    }

    Measurement Measurement::Matrix(MatXd m, const Unit& u)
    {
        Measurement mm;
        const Index rows = m.rows();
        const Index cols = m.cols();
        mm.storage_ = std::move(m);
        mm.data_type_   = DataType::kReal;
        mm.unit_        = u;
        mm.shape_.push_back(rows);
        mm.shape_.push_back(cols);
        return mm;
    }

    Measurement Measurement::Matrix(MatXi m, const Unit& u)
    {
        Measurement mm;
        const Index rows = m.rows();
        const Index cols = m.cols();
        mm.storage_ = std::move(m);
        mm.data_type_   = DataType::kInteger;
        mm.unit_        = u;
        mm.shape_.push_back(rows);
        mm.shape_.push_back(cols);
        return mm;
    }

    Measurement Measurement::Matrix(MatXcd m, const Unit& u)
    {
        Measurement mm;
        const Index rows = m.rows();
        const Index cols = m.cols();
        mm.storage_ = std::move(m);
        mm.data_type_   = DataType::kComplex;
        mm.unit_        = u;
        mm.shape_.push_back(rows);
        mm.shape_.push_back(cols);
        return mm;
    }

    Measurement Measurement::Matrix(MatConstMap<double> m, const Unit& u)
    {
        return Measurement::Matrix(MatXd(m), u);
    }

    Measurement Measurement::Matrix(MatConstMap<int> m, const Unit& u)
    {
        return Measurement::Matrix(MatXi(m), u);
    }

    Measurement Measurement::Matrix(MatConstMap<std::complex<double> > m,
                                    const Unit& u)
    {
        return Measurement::Matrix(MatXcd(m), u);
    }

    Measurement Measurement::Matrix(const MatXs& m)
    {
        Measurement mm;
        mm.storage_ = m;
        mm.data_type_   = DataType::kString;
        mm.shape_.push_back(m.dimension(0));
        mm.shape_.push_back(m.dimension(1));
        return mm;
    }

    // =========================================================================
    // Measurement -- queries
    // =========================================================================

    bool Measurement::has_value() const
    {
        // Default-constructed Measurement is 0.0 real scalar -- always has a value.
        // (variant is never empty.)
        return true;
    }

    std::string Measurement::to_string() const
    {
        // No formatter object is constructed: Format() is a free function and
        // the options are read straight from the process-wide defaults.
        return Format(*this, FormatDefaults::Instance().Get());
    }

    std::string Measurement::to_string(const FormatOptions& options) const
    {
        return Format(*this, options);
    }

    bool Measurement::operator==(const Measurement& other) const
    {
        MeasurementEqualityVisitor v(&other.storage_);
        return boost::apply_visitor(v, storage_);
    }

    // =========================================================================
    // Measurement -- element_at (vector -> scalar / matrix -> scalar)
    // =========================================================================

    Measurement Measurement::element_at(Index i) const
    {
        if (shape_.kind() != DataKind::kVector)
            throw std::logic_error("element_at(Index) requires vector data");
        switch (data_type_)
        {
            case DataType::kReal:    return Measurement::Real(boost::get<VecXd>(storage_)(i), unit_);
            case DataType::kInteger: return Measurement::Integer(boost::get<VecXi>(storage_)(i), unit_);
            case DataType::kComplex: return Measurement::Complex(boost::get<VecXcd>(storage_)(i), unit_);
            case DataType::kString:  return Measurement::String(boost::get<VecXs>(storage_)(i));
            default: break;  // kBoolean is scalar-only
        }
        throw std::logic_error("unsupported dtype");
    }

    Measurement Measurement::element_at(Index r, Index c) const
    {
        if (shape_.kind() != DataKind::kMatrix)
            throw std::logic_error("element_at(Index, Index) requires matrix data");
        switch (data_type_)
        {
            case DataType::kReal:    return Measurement::Real(boost::get<MatXd>(storage_)(r, c), unit_);
            case DataType::kInteger: return Measurement::Integer(boost::get<MatXi>(storage_)(r, c), unit_);
            case DataType::kComplex: return Measurement::Complex(boost::get<MatXcd>(storage_)(r, c), unit_);
            case DataType::kString:  return Measurement::String(boost::get<MatXs>(storage_)(r, c));
            default: break;  // kBoolean is scalar-only
        }
        throw std::logic_error("unsupported dtype");
    }

    // =========================================================================
    // Measurement -- at (slicing via MultiIndexSelector)
    // =========================================================================

    Measurement Measurement::at(const std::vector<MultiIndexSelector>& selectors) const
    {
        if (shape_.kind() == DataKind::kScalar)
            throw std::logic_error("at is invalid for scalar Measurement");

        const std::size_t ndim = (shape_.kind() == DataKind::kVector) ? 1 : 2;
        if (selectors.size() > ndim)
            throw std::invalid_argument("too many selectors for Measurement::at");

        // Pad short selectors with Any()
        std::vector<MultiIndexSelector> padded = selectors;
        while (padded.size() < ndim)
            padded.push_back(MultiIndexSelector::Any());

        if (shape_.kind() == DataKind::kVector)
        {
            const std::vector<Index> selected = padded[0].resolve(shape_[0]);

            if (selected.size() == 1)
                return element_at(selected[0]);

            // Sub-vector: extract selected elements
            switch (data_type_)
            {
                case DataType::kReal: {
                    const auto& src = boost::get<VecXd>(storage_);
                    VecXd dst(static_cast<Index>(selected.size()));
                    for (std::size_t i = 0; i < selected.size(); ++i)
                        dst(static_cast<Index>(i)) = src(selected[i]);
                    return Measurement::Vector(dst).set_unit(unit_);
                }
                case DataType::kInteger: {
                    const auto& src = boost::get<VecXi>(storage_);
                    VecXi dst(static_cast<Index>(selected.size()));
                    for (std::size_t i = 0; i < selected.size(); ++i)
                        dst(static_cast<Index>(i)) = src(selected[i]);
                    return Measurement::Vector(dst).set_unit(unit_);
                }
                case DataType::kComplex: {
                    const auto& src = boost::get<VecXcd>(storage_);
                    VecXcd dst(static_cast<Index>(selected.size()));
                    for (std::size_t i = 0; i < selected.size(); ++i)
                        dst(static_cast<Index>(i)) = src(selected[i]);
                    return Measurement::Vector(dst).set_unit(unit_);
                }
                case DataType::kString: {
                    const auto& src = boost::get<VecXs>(storage_);
                    VecXs dst(static_cast<Index>(selected.size()));
                    for (std::size_t i = 0; i < selected.size(); ++i)
                        dst(static_cast<Index>(i)) = src(selected[i]);
                    return Measurement::Vector(dst).set_unit(unit_);
                }
                default: break;
            }
            throw std::logic_error("unsupported dtype in Measurement::at");
        }

        // Matrix
        const std::vector<Index> sel_rows = padded[0].resolve(shape_[0]);
        const std::vector<Index> sel_cols = padded[1].resolve(shape_[1]);

        if (sel_rows.size() == 1 && sel_cols.size() == 1)
            return element_at(sel_rows[0], sel_cols[0]);

        if (sel_rows.size() == 1 || sel_cols.size() == 1)
        {
            // Single row or single column -> vector
            const bool single_row = (sel_rows.size() == 1);
            const std::vector<Index>& remaining = single_row ? sel_cols : sel_rows;
            const Index width = static_cast<Index>(remaining.size());

            switch (data_type_)
            {
                case DataType::kReal: {
                    const auto& src = boost::get<MatXd>(storage_);
                    VecXd dst(width);
                    for (Index i = 0; i < width; ++i) {
                        Index r = single_row ? sel_rows[0] : remaining[i];
                        Index c = single_row ? remaining[i] : sel_cols[0];
                        dst(i) = src(r, c);
                    }
                    return Measurement::Vector(dst).set_unit(unit_);
                }
                case DataType::kInteger: {
                    const auto& src = boost::get<MatXi>(storage_);
                    VecXi dst(width);
                    for (Index i = 0; i < width; ++i) {
                        Index r = single_row ? sel_rows[0] : remaining[i];
                        Index c = single_row ? remaining[i] : sel_cols[0];
                        dst(i) = src(r, c);
                    }
                    return Measurement::Vector(dst).set_unit(unit_);
                }
                case DataType::kComplex: {
                    const auto& src = boost::get<MatXcd>(storage_);
                    VecXcd dst(width);
                    for (Index i = 0; i < width; ++i) {
                        Index r = single_row ? sel_rows[0] : remaining[i];
                        Index c = single_row ? remaining[i] : sel_cols[0];
                        dst(i) = src(r, c);
                    }
                    return Measurement::Vector(dst).set_unit(unit_);
                }
                case DataType::kString: {
                    const auto& src = boost::get<MatXs>(storage_);
                    VecXs dst(width);
                    for (Index i = 0; i < width; ++i) {
                        Index r = single_row ? sel_rows[0] : remaining[i];
                        Index c = single_row ? remaining[i] : sel_cols[0];
                        dst(i) = src(r, c);
                    }
                    return Measurement::Vector(dst).set_unit(unit_);
                }
                default: break;
            }
            throw std::logic_error("unsupported dtype in Measurement::at");
        }

        // Sub-matrix: extract selected rows x columns
        switch (data_type_)
        {
            case DataType::kReal: {
                const auto& src = boost::get<MatXd>(storage_);
                MatXd dst(static_cast<Index>(sel_rows.size()),
                          static_cast<Index>(sel_cols.size()));
                for (Index i = 0; i < static_cast<Index>(sel_rows.size()); ++i)
                    for (Index j = 0; j < static_cast<Index>(sel_cols.size()); ++j)
                        dst(i, j) = src(sel_rows[i], sel_cols[j]);
                return Measurement::Matrix(dst).set_unit(unit_);
            }
            case DataType::kInteger: {
                const auto& src = boost::get<MatXi>(storage_);
                MatXi dst(static_cast<Index>(sel_rows.size()),
                          static_cast<Index>(sel_cols.size()));
                for (Index i = 0; i < static_cast<Index>(sel_rows.size()); ++i)
                    for (Index j = 0; j < static_cast<Index>(sel_cols.size()); ++j)
                        dst(i, j) = src(sel_rows[i], sel_cols[j]);
                return Measurement::Matrix(dst).set_unit(unit_);
            }
            case DataType::kComplex: {
                const auto& src = boost::get<MatXcd>(storage_);
                MatXcd dst(static_cast<Index>(sel_rows.size()),
                           static_cast<Index>(sel_cols.size()));
                for (Index i = 0; i < static_cast<Index>(sel_rows.size()); ++i)
                    for (Index j = 0; j < static_cast<Index>(sel_cols.size()); ++j)
                        dst(i, j) = src(sel_rows[i], sel_cols[j]);
                return Measurement::Matrix(dst).set_unit(unit_);
            }
            case DataType::kString: {
                const auto& src = boost::get<MatXs>(storage_);
                MatXs dst(static_cast<Index>(sel_rows.size()),
                          static_cast<Index>(sel_cols.size()));
                for (Index i = 0; i < static_cast<Index>(sel_rows.size()); ++i)
                    for (Index j = 0; j < static_cast<Index>(sel_cols.size()); ++j)
                        dst(i, j) = src(sel_rows[i], sel_cols[j]);
                return Measurement::Matrix(dst).set_unit(unit_);
            }
            default: break;
        }
        throw std::logic_error("unsupported dtype in Measurement::at");
    }

    // =========================================================================
    //  Rendering
    // =========================================================================
    //
    //  Layered, all free functions taking (unit, options) explicitly:
    //
    //    FormatElement / FormatElementInt   pure NUMBER layer -- no unit
    //    ResolveScale                       the only expensive step
    //    FormatValue / FormatValueInt / FormatComplex
    //    RenderScalar / RenderVector / RenderMatrix
    //    Format                             dispatch on cached kind / dtype
    //
    //  Nothing is stored, so there is no object to construct and no state to
    //  make thread-local: Measurement::to_string() just calls Format().

    namespace
    {
        /// Append @p v to @p out using the same formatting std::ostream's
        /// default `operator<<(double)` produces (6 significant digits,
        /// shortest of fixed / scientific, no trailing zeros).
        ///
        /// std::ostringstream is the single biggest cost in the old
        /// implementation: constructing one pulls in the global locale and a
        /// streambuf, and each `<<` goes through the num_put facet.  snprintf
        /// with "%.6g" is byte-for-byte equivalent for every finite double and
        /// roughly an order of magnitude faster.
        void AppendDouble(std::string& out, double v)
        {
            char buf[40];
            const int n = std::snprintf(buf, sizeof(buf), "%.6g", v);
            if (n > 0)
                out.append(buf, static_cast<std::size_t>(n));
        }

        void AppendInt(std::string& out, int v)
        {
            char buf[24];
            const int n = std::snprintf(buf, sizeof(buf), "%d", v);
            if (n > 0)
                out.append(buf, static_cast<std::size_t>(n));
        }

        /// Clamp significant digits to a sane range (a double round-trips in
        /// at most 17 digits).
        int ClampDigits(int d)
        {
            if (d < 1) return 1;
            if (d > 17) return 17;
            return d;
        }

        /// Render @p v with @p digits significant digits, shortest of fixed /
        /// scientific -- i.e. printf's "%.<digits>g".
        void AppendSignificant(std::string& out, double v, int digits)
        {
            char buf[64];
            char fmt[16];
            std::snprintf(fmt, sizeof(fmt), "%%.%dg", ClampDigits(digits));
            const int n = std::snprintf(buf, sizeof(buf), fmt, v);
            if (n > 0)
                out.append(buf, static_cast<std::size_t>(n));
        }

        /// Render @p v in fixed (non-exponent) notation with @p digits
        /// significant digits, WITHOUT trailing zeros.
        ///
        /// "Full" means: never use an exponent -- every digit before the
        /// decimal point is shown.  @p digits caps precision but does not
        /// PAD: the budget is spent on the integer digits first and whatever
        /// remains becomes decimals, and only as many of those as are
        /// actually significant are kept.
        ///
        ///   1530000.123 w/ 6 -> "1530000"   (7 int digits already exceed it)
        ///   1000.0      w/ 6 -> "1000"      (not "1000.00")
        ///   0.002       w/ 6 -> "0.002"     (not "0.00200000")
        ///   1234.567890 w/ 6 -> "1234.57"
        ///   1234.567890 w/ 3 -> "1235"
        void AppendFull(std::string& out, double v, int digits)
        {
            const int d = ClampDigits(digits);
            const double av = std::abs(v);
            if (av == 0.0)
            {
                out += '0';
                return;
            }
            // Digits before the decimal point.  This is <= 0 for values below
            // 1 (0.002 -> -2), which correctly gives such values a larger
            // decimal budget so they still get @p d significant digits.
            const int int_digits = static_cast<int>(std::floor(std::log10(av))) + 1;
            int decimals = d - int_digits;
            if (decimals < 0) decimals = 0;
            if (decimals > 17) decimals = 17;

            // "%.0f" of a huge double expands to every integer digit (1e308
            // needs 309 of them), so snprintf's return value -- which counts
            // what it WOULD have written -- can exceed the buffer.  Clamp to
            // what was actually stored, or we would read past the end.
            char buf[64];
            char fmt[16];
            std::snprintf(fmt, sizeof(fmt), "%%.%df", decimals);
            const int n = std::snprintf(buf, sizeof(buf), fmt, v);
            if (n <= 0)
                return;
            int len = (n < static_cast<int>(sizeof(buf))) ? n
                                                          : static_cast<int>(sizeof(buf)) - 1;

            // Drop trailing zeros (and a now-bare decimal point) so "1000.00"
            // prints as "1000" and "3.14000" as "3.14".  Only touch the
            // fractional part -- an integer like "1000" has no '.' and is
            // emitted verbatim.
            const char* dot = std::strchr(buf, '.');
            if (dot != nullptr)
            {
                while (len > 0 && buf[len - 1] == '0')
                    --len;
                if (len > 0 && buf[len - 1] == '.')
                    --len;
            }
            out.append(buf, static_cast<std::size_t>(len));
        }

        /// Render an exact integer in "Full" notation: no decimal point at
        /// all.  An integer has no fractional part, so printing "42.0000"
        /// (what AppendFull would do with a 6-digit budget) is wrong.
        ///
        /// Takes `int` -- the same type the Storage variant holds for
        /// DataType::kInteger (and the element type of VecXi / MatXi).
        void AppendFullInt(std::string& out, int v)
        {
            char buf[24];
            const int n = std::snprintf(buf, sizeof(buf), "%d", v);
            if (n > 0)
                out.append(buf, static_cast<std::size_t>(n));
        }

        /// Render @p v in scientific notation: mantissa with @p digits
        /// significant digits, then "e<exp>" (1000 -> "1e3").
        void AppendScientific(std::string& out, double v, int digits)
        {
            const int d = ClampDigits(digits);
            if (v == 0.0)
            {
                out += "0e0";
                return;
            }
            const int exp = static_cast<int>(std::floor(std::log10(std::abs(v))));
            const double mant = v / std::pow(10.0, exp);
            char buf[64];
            char fmt[16];
            std::snprintf(fmt, sizeof(fmt), "%%.%dg", d - 1);
            const int n = std::snprintf(buf, sizeof(buf), fmt, mant);
            if (n > 0)
                out.append(buf, static_cast<std::size_t>(n));
            out += 'e';
            AppendInt(out, exp);
        }

        /// Render @p v as an exact integer in base @p base, with @p prefix.
        /// Non-integral / out-of-range values fall back to plain decimal
        /// (a positional base cannot represent a fraction).
        void AppendIntegerBase(std::string& out, double v, int base,
                               const char* prefix, int digits)
        {
            // Only exact integers are representable in a positional base.
            if (!std::isfinite(v) || v != std::floor(v) ||
                std::abs(v) > 9.0e15)
            {
                AppendFull(out, v, digits);
                return;
            }
            const bool neg = v < 0.0;
            unsigned long long n =
                static_cast<unsigned long long>(neg ? -v : v);
            if (neg) out += '-';
            out += prefix;
            if (n == 0)
            {
                out += '0';
                return;
            }
            char buf[80];
            int pos = 0;
            while (n > 0)
            {
                const int digit = static_cast<int>(n % static_cast<unsigned>(base));
                buf[pos++] = (digit < 10) ? char('0' + digit)
                                          : char('a' + (digit - 10));
                n /= static_cast<unsigned>(base);
            }
            while (pos > 0) out += buf[--pos];
        }
    } // namespace

    std::string WithUnit(const std::string& s, const Unit& unit_,
                         const FormatOptions& options_)
    {
        // The unit suffix is suppressed entirely when show_unit is false --
        // including the dimensionless scale prefix, which is part of the
        // suffix rather than a unit of its own.
        if (!options_.show_unit || !unit_.has_dimension())
            return s;
        return s + " " + unit_.to_string();
    }

    // --- number rendering ------------------------------------------------

    /// Render one number according to @p options.number_format.  This is the
    /// pure NUMBER layer: it knows nothing about units -- no scaling and no
    /// suffix (the caller does both once, see FormatValue()).
    std::string FormatElement(double v, const FormatOptions& options_)
    {
        std::string out;
        out.reserve(32);
        switch (options_.number_format)
        {
        case NumberFormat::kFull:
            AppendFull(out, v, options_.significant_digits);
            break;
        case NumberFormat::kScientific:
            AppendScientific(out, v, options_.significant_digits);
            break;
        case NumberFormat::kEngineering:
            // The exponent is folded into the unit prefix by format_value();
            // here we only render the mantissa.
            AppendSignificant(out, v, options_.significant_digits);
            break;
        case NumberFormat::kHex:
            AppendIntegerBase(out, v, 16, "0x", options_.significant_digits);
            break;
        case NumberFormat::kOctal:
            AppendIntegerBase(out, v, 8, "0", options_.significant_digits);
            break;
        case NumberFormat::kBinary:
            AppendIntegerBase(out, v, 2, "0b", options_.significant_digits);
            break;
        }
        return out;
    }

    /// Integer variant of FormatElement(): keeps the value integral so that
    /// "Full" renders 42 as "42" rather than "42.0000".
    ///
    /// Takes `int` -- the same type the Storage variant holds for
    /// DataType::kInteger (and the element type of VecXi / MatXi).
    std::string FormatElementInt(int v, const FormatOptions& options_)
    {
        std::string out;
        out.reserve(32);
        switch (options_.number_format)
        {
        case NumberFormat::kFull:
            AppendFullInt(out, v);
            break;
        case NumberFormat::kScientific:
            AppendScientific(out, static_cast<double>(v),
                             options_.significant_digits);
            break;
        case NumberFormat::kEngineering:
            AppendSignificant(out, static_cast<double>(v),
                              options_.significant_digits);
            break;
        case NumberFormat::kHex:
            AppendIntegerBase(out, static_cast<double>(v), 16, "0x",
                              options_.significant_digits);
            break;
        case NumberFormat::kOctal:
            AppendIntegerBase(out, static_cast<double>(v), 8, "0",
                              options_.significant_digits);
            break;
        case NumberFormat::kBinary:
            AppendIntegerBase(out, static_cast<double>(v), 2, "0b",
                              options_.significant_digits);
            break;
        }
        return out;
    }

    /// Resolve how a value must be scaled and which suffix (if any) to append.
    ///
    /// Shared by FormatValue() and FormatComplex() so a complex value obeys
    /// exactly the same number_format / show_unit rules as a real one.
    ///
    /// The suffix is stored PRE-FORMATTED: it already carries the leading
    /// space for a unit (" GHz") or none for a SPICE prefix ("G"), so the
    /// caller just appends it.
    DisplayScale ResolveScale(double v, const Unit& unit_,
                              const FormatOptions& options_)
    {
        DisplayScale d;

        // Hex / octal / binary are exact integer representations: no unit
        // scaling and no unit suffix apply.
        if (options_.number_format == NumberFormat::kHex ||
            options_.number_format == NumberFormat::kOctal ||
            options_.number_format == NumberFormat::kBinary)
        {
            return d;
        }

        if (!options_.show_unit &&
            options_.number_format == NumberFormat::kEngineering)
        {
            // Engineering + unit hidden: the 10^3 step is part of the MODE,
            // not of the unit, so the scaling still applies -- but only the
            // prefix is shown, SPICE-style with no space: 2.4e9 Hz -> "2.4G".
            const UnitScale bd = unit_.best_display(v);
            d.scale = bd.scale;
            d.suffix = bd.prefix;
        }
        else if (!options_.show_unit)
        {
            // Unit hidden in any other mode: render the raw value.  No
            // auto-scaling either -- the scale is carried by the unit prefix,
            // so without the unit there is no way to express it
            // (0.002 V -> "0.002", not "2").
            d.scale = 1.0;
        }
        else if (options_.number_format == NumberFormat::kFull ||
                 options_.number_format == NumberFormat::kScientific)
        {
            // "Full" and "Scientific" show the number as-is: the exponent is
            // carried by the digits (or by "e<n>"), so the unit must NOT be
            // auto-scaled.  1530000.123 Hz -> "1530000 Hz", not "1.53 MHz".
            d.scale = 1.0;
            const std::string name = unit_.to_string();
            if (!name.empty())
            {
                d.suffix = ' ' + name;
            }
        }
        else
        {
            // kEngineering: auto-scale to a readable prefix.
            //
            // This applies to DIMENSIONLESS values too: a bare scale prefix
            // ("m", "M", "K", "G" ...) is not a *unit*, but it is exactly
            // what "best unit display" means here -- 0.002 renders as "2 m".
            const UnitScale bd = unit_.best_display(v);
            d.scale = bd.scale;
            if (!bd.name.empty())
            {
                d.suffix = ' ' + bd.name;
            }
        }
        return d;
    }

    /// Append the resolved suffix to @p out (it is already pre-formatted).
    void AppendSuffix(std::string& out, const DisplayScale& d)
    {
        out += d.suffix;
    }

    /// Render one number using an ALREADY-RESOLVED scale.  Cheap: no unit
    /// lookup, just the number formatting plus the suffix.
    std::string FormatWithScale(double v, const DisplayScale& scale,
                                NumberFormat format, int digits)
    {
        if (!std::isfinite(v)) return "<invalid>";
        FormatOptions o;
        o.number_format = format;
        o.significant_digits = digits;
        std::string out = FormatElement(v * scale.scale, o);
        out += scale.suffix;
        return out;
    }

    /// Render one integer using an already-resolved scale.  An integer never
    /// gets a decimal point in kFull mode.
    std::string FormatWithScale(int v, const DisplayScale& scale,
                                NumberFormat format, int digits)
    {
        FormatOptions o;
        o.number_format = format;
        o.significant_digits = digits;
        const double scaled = static_cast<double>(v) * scale.scale;
        std::string out;
        const bool integral = (scaled == std::floor(scaled)) &&
                              scaled >= static_cast<double>(INT_MIN) &&
                              scaled <= static_cast<double>(INT_MAX);
        if (integral)
        {
            out = FormatElementInt(static_cast<int>(scaled), o);
        }
        else
        {
            out = FormatElement(scaled, o);
        }
        out += scale.suffix;
        return out;
    }

    /// Integer variant of FormatValue(): applies the unit scale, then renders
    /// the result as an integer when the scale left it integral (e.g. Full
    /// with no scaling), and as a double otherwise (Engineering may turn
    /// 4700 Ohm into 4.7 KOhm).
    std::string FormatValueInt(int v, const Unit& unit_,
                               const FormatOptions& options_)
    {
        if (options_.number_format == NumberFormat::kHex ||
            options_.number_format == NumberFormat::kOctal ||
            options_.number_format == NumberFormat::kBinary)
        {
            return FormatElementInt(v, options_);
        }

        const DisplayScale d = ResolveScale(static_cast<double>(v), unit_, options_);
        const double scaled = static_cast<double>(v) * d.scale;

        std::string out;
        // Render as an integer only when the scale left the value integral
        // AND it still fits in int (Engineering may scale 4700 Ohm to 4.7,
        // which must fall back to the double path).
        const bool integral = (scaled == std::floor(scaled)) &&
                              scaled >= static_cast<double>(INT_MIN) &&
                              scaled <= static_cast<double>(INT_MAX);
        if (integral)
        {
            out = FormatElementInt(static_cast<int>(scaled), options_);
        }
        else
        {
            out = FormatElement(scaled, options_);
        }
        AppendSuffix(out, d);
        return out;
    }

    /// Render one scalar: apply the unit scale, format the number, and append
    /// the unit suffix.
    std::string FormatValue(double v, const Unit& unit_,
                            const FormatOptions& options_)
    {
        if (!std::isfinite(v)) return "<invalid>";

        // Hex / octal / binary are exact integer representations: no unit
        // scaling and no unit suffix apply.
        if (options_.number_format == NumberFormat::kHex ||
            options_.number_format == NumberFormat::kOctal ||
            options_.number_format == NumberFormat::kBinary)
        {
            return FormatElement(v, options_);
        }

        const DisplayScale d = ResolveScale(v, unit_, options_);
        std::string out = FormatElement(v * d.scale, options_);
        AppendSuffix(out, d);
        return out;
    }

    /// Render a complex value according to options_.complex_format.
    std::string FormatComplex(const std::complex<double>& v, const Unit& unit_,
                              const FormatOptions& options_)
    {
        if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) return "<invalid>";

        std::string out;
        out.reserve(64);

        switch (options_.complex_format)
        {
        case ComplexFormat::kRealImaginary:
        {
            // Both parts share ONE unit scale, resolved from the magnitude so
            // that a value like 0.002+0.003i is not scaled by the (possibly
            // near-zero) real part alone.  ResolveScale() applies exactly the
            // same number_format / show_unit rules as FormatValue().
            const DisplayScale d = ResolveScale(std::abs(v), unit_, options_);

            // FormatElement() honours number_format, so Full / Scientific /
            // Hex / ... apply to complex values too (they used to be ignored).
            out += FormatElement(v.real() * d.scale, options_);
            const double im = v.imag() * d.scale;
            if (im >= 0.0) out += '+';
            out += FormatElement(im, options_);
            out += 'i';
            AppendSuffix(out, d);
            return out;
        }
        case ComplexFormat::kMagDegrees:
        case ComplexFormat::kMagRadians:
        {
            // Pure "a/b" pair: magnitude and phase, NO unit and NO scaling.
            // A magnitude/phase pair is not a value in the Measurement's unit
            // the way a real part is, so the unit is dropped entirely.
            AppendSignificant(out, std::abs(v), options_.significant_digits);
            out += '/';
            const double phase = std::arg(v) * ((options_.complex_format ==
                                                 ComplexFormat::kMagDegrees)
                                                    ? 180.0 / M_PI
                                                    : 1.0);
            AppendSignificant(out, phase, options_.significant_digits);
            return out;
        }
        case ComplexFormat::kDbDegrees:
        case ComplexFormat::kDbRadians:
        {
            // Same: dB magnitude / phase, no unit, no scaling.
            const double mag = std::abs(v);
            const double db = (mag > 0.0) ? 20.0 * std::log10(mag) : -999.0;
            AppendSignificant(out, db, options_.significant_digits);
            out += '/';
            const double phase = std::arg(v) * ((options_.complex_format ==
                                                 ComplexFormat::kDbDegrees)
                                                    ? 180.0 / M_PI
                                                    : 1.0);
            AppendSignificant(out, phase, options_.significant_digits);
            return out;
        }
        }
        return out;
    }

    // --- scalar ------------------------------------------------------------

    std::string RenderScalar(double v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        return FormatValue(v, unit_, options_);
    }

    std::string RenderScalar(int v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        // Integer path: keeps 42 as "42" instead of "42.0000".
        return FormatValueInt(v, unit_, options_);
    }

    std::string RenderScalar(const std::complex<double>& v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        return FormatComplex(v, unit_, options_);
    }

    std::string RenderScalar(const std::string& v, const Unit&,
                             const FormatOptions&)
    {
        return v;
    }

    std::string RenderScalar(bool v, const Unit&, const FormatOptions&)
    {
        return v ? "TRUE" : "FALSE";
    }

    // -- vector --------------------------------------------------------------

    std::string RenderVector(const VecXd& v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        std::string out;
        out.reserve(static_cast<std::size_t>(v.size()) * 12 + 2);
        out += '[';
        for (Index i = 0; i < v.size(); ++i)
        {
            if (i > 0) out += ',';
            out += FormatElement(v(i), options_);
        }
        out += ']';
        return WithUnit(out, unit_, options_);
    }

    std::string RenderVector(const VecXi& v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        std::string out;
        out.reserve(static_cast<std::size_t>(v.size()) * 8 + 2);
        out += '[';
        for (Index i = 0; i < v.size(); ++i)
        {
            if (i > 0) out += ',';
            out += FormatElementInt(v(i), options_);
        }
        out += ']';
        return WithUnit(out, unit_, options_);
    }

    std::string RenderVector(const VecXcd& v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        std::string out;
        out.reserve(static_cast<std::size_t>(v.size()) * 24 + 2);
        out += '[';
        for (Index i = 0; i < v.size(); ++i)
        {
            if (i > 0) out += ',';
            out += FormatComplex(v(i), unit_, options_);
        }
        out += ']';
        return out;
    }

    std::string RenderVector(const VecXs& v, const Unit&,
                             const FormatOptions&)
    {
        std::string out;
        out += '[';
        for (Index i = 0; i < v.dimension(0); ++i)
        {
            if (i > 0) out += ',';
            out += v(i);
        }
        out += ']';
        return out;
    }

    // -- matrix --------------------------------------------------------------

    std::string RenderMatrix(const MatXd& v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        std::string out;
        out.reserve(static_cast<std::size_t>(v.size()) * 12 + 4);
        out += '[';
        for (Index r = 0; r < v.rows(); ++r)
        {
            if (r > 0) out += ',';
            out += '[';
            for (Index c = 0; c < v.cols(); ++c)
            {
                if (c > 0) out += ',';
                out += FormatElement(v(r, c), options_);
            }
            out += ']';
        }
        out += ']';
        return WithUnit(out, unit_, options_);
    }

    std::string RenderMatrix(const MatXi& v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        std::string out;
        out.reserve(static_cast<std::size_t>(v.size()) * 8 + 4);
        out += '[';
        for (Index r = 0; r < v.rows(); ++r)
        {
            if (r > 0) out += ',';
            out += '[';
            for (Index c = 0; c < v.cols(); ++c)
            {
                if (c > 0) out += ',';
                out += FormatElementInt(v(r, c), options_);
            }
            out += ']';
        }
        out += ']';
        return WithUnit(out, unit_, options_);
    }

    std::string RenderMatrix(const MatXcd& v, const Unit& unit_,
                             const FormatOptions& options_)
    {
        std::string out;
        out.reserve(static_cast<std::size_t>(v.size()) * 24 + 4);
        out += '[';
        for (Index r = 0; r < v.rows(); ++r)
        {
            if (r > 0) out += ',';
            out += '[';
            for (Index c = 0; c < v.cols(); ++c)
            {
                if (c > 0) out += ',';
                out += FormatComplex(v(r, c), unit_, options_);
            }
            out += ']';
        }
        out += ']';
        return out;
    }

    std::string RenderMatrix(const MatXs& v, const Unit&,
                             const FormatOptions&)
    {
        std::string out;
        out += '[';
        for (Index r = 0; r < v.dimension(0); ++r)
        {
            if (r > 0) out += ',';
            out += '[';
            for (Index c = 0; c < v.dimension(1); ++c)
            {
                if (c > 0) out += ',';
                out += v(r, c);
            }
            out += ']';
        }
        out += ']';
        return out;
    }

    // =====================================================================
    // FormatDefaults / FormatScope -- the process-wide default options.
    //
    // A plain static (NOT thread_local): the options are read-only during
    // rendering and the unit is a function parameter, so there is no per-call
    // mutable state and nothing to make thread-local.
    // =====================================================================

    FormatDefaults& FormatDefaults::Instance()
    {
        static FormatDefaults instance;
        return instance;
    }

    void FormatDefaults::Set(const FormatOptions& options)
    {
        options_ = options;
    }

    FormatScope::FormatScope(const FormatOptions& options)
        : saved_(FormatDefaults::Instance().Get())
    {
        FormatDefaults::Instance().Set(options);
    }

    FormatScope::~FormatScope()
    {
        FormatDefaults::Instance().Set(saved_);
    }

    // =====================================================================
    // Format -- the general entry point.
    //
    // Dispatches on the CACHED data_type_ / shape instead of
    // boost::apply_visitor: Measurement already knows which alternative is
    // active, so no visitor object is constructed and the call is a direct
    // (often inlined) function call rather than a 12-way runtime dispatch.
    // =====================================================================

    std::string Format(const Measurement& m, const FormatOptions& options)
    {
        const Unit& unit = m.unit();
        const Measurement::Storage& storage = m.storage();

        switch (m.data_kind())
        {
        case DataKind::kScalar:
            switch (m.data_type())
            {
            case DataType::kReal:
                return RenderScalar(boost::get<double>(storage), unit, options);
            case DataType::kInteger:
                return RenderScalar(boost::get<int>(storage), unit, options);
            case DataType::kComplex:
                return RenderScalar(boost::get<std::complex<double>>(storage),
                                    unit, options);
            case DataType::kString:
                return RenderScalar(boost::get<std::string>(storage), unit, options);
            case DataType::kBoolean:
                return RenderScalar(boost::get<bool>(storage), unit, options);
            }
            return std::string();

        case DataKind::kVector:
            switch (m.data_type())
            {
            case DataType::kReal:
                return RenderVector(boost::get<VecXd>(storage), unit, options);
            case DataType::kInteger:
                return RenderVector(boost::get<VecXi>(storage), unit, options);
            case DataType::kComplex:
                return RenderVector(boost::get<VecXcd>(storage), unit, options);
            case DataType::kString:
                return RenderVector(boost::get<VecXs>(storage), unit, options);
            default:
                break;   // kBoolean is scalar-only
            }
            return std::string();

        case DataKind::kMatrix:
            switch (m.data_type())
            {
            case DataType::kReal:
                return RenderMatrix(boost::get<MatXd>(storage), unit, options);
            case DataType::kInteger:
                return RenderMatrix(boost::get<MatXi>(storage), unit, options);
            case DataType::kComplex:
                return RenderMatrix(boost::get<MatXcd>(storage), unit, options);
            case DataType::kString:
                return RenderMatrix(boost::get<MatXs>(storage), unit, options);
            default:
                break;   // kBoolean is scalar-only
            }
            return std::string();
        }
        return std::string();
    }

// =========================================================================
//  Measurement -> converted_to / canonicalized
// =========================================================================

Measurement Measurement::converted_to(const Unit& target) const {
    if (unit_.has_dimension() && target.has_dimension() &&
        !unit_.same_dimension(target)) {
        throw std::invalid_argument(
            "Measurement::converted_to: dimension mismatch [" +
            unit_.to_string() + "] -> [" + target.to_string() + "]");
    }

    // Strings / booleans carry no numeric value -- only re-tag the unit.
    if (data_type_ == DataType::kString ||
        data_type_ == DataType::kBoolean) {
        Measurement result(*this);
        result.unit_ = target;
        return result;
    }

    const double factor = unit_.multiplier() / target.multiplier();

    // Fast path: no scaling needed (also covers target == unit()).
    if (factor == 1.0) {
        Measurement result(*this);
        result.unit_ = target;
        return result;
    }

    // Integer storage cannot hold scaled values (int * 1e-3 truncates to 0);
    // promote to real first.
    const DataType res_dtype =
        (data_type_ == DataType::kInteger) ? DataType::kReal : data_type_;

    Measurement result;
    result.data_type_ = res_dtype;
    result.shape_     = shape_;
    result.unit_      = target;

    const DataKind kind = shape_.kind();

    if (kind == DataKind::kScalar) {
        // Dispatch on dtype BEFORE reading storage: a kComplex measurement
        // must not be read via boost::get<double>() (throws bad_get).
        if (res_dtype == DataType::kComplex) {
            std::complex<double> cv = boost::get<std::complex<double> >(storage_);
            cv *= factor;
            result.storage_ = cv;
        } else {
            const double v = (data_type_ == DataType::kInteger)
                             ? static_cast<double>(boost::get<int>(storage_))
                             : boost::get<double>(storage_);
            result.storage_ = v * factor;
        }
        return result;
    }

    if (kind == DataKind::kVector) {
        if (res_dtype == DataType::kComplex) {
            VecXcd vec = boost::get<VecXcd>(storage_);
            vec *= factor;
            result.storage_ = vec;
        } else if (data_type_ == DataType::kInteger) {
            const VecXi& src = boost::get<VecXi>(storage_);
            VecXd vec(src.size());
            for (Index i = 0; i < src.size(); ++i)
                vec(i) = static_cast<double>(src(i)) * factor;
            result.storage_ = vec;
        } else {
            VecXd vec = boost::get<VecXd>(storage_);
            vec *= factor;
            result.storage_ = vec;
        }
        return result;
    }

    // Matrix
    if (res_dtype == DataType::kComplex) {
        MatXcd mat = boost::get<MatXcd>(storage_);
        mat *= factor;
        result.storage_ = mat;
    } else if (data_type_ == DataType::kInteger) {
        const MatXi& src = boost::get<MatXi>(storage_);
        MatXd mat(src.rows(), src.cols());
        for (Index r = 0; r < src.rows(); ++r)
            for (Index c = 0; c < src.cols(); ++c)
                mat(r, c) = static_cast<double>(src(r, c)) * factor;
        result.storage_ = mat;
    } else {
        MatXd mat = boost::get<MatXd>(storage_);
        mat *= factor;
        result.storage_ = mat;
    }
    return result;
}

Measurement Measurement::canonicalized() const {
    return converted_to(unit_.canonicalized());
}

bool Measurement::is_canonicalized() const {
    return unit_.is_canonical();
}

void Measurement::canonicalize() {
    *this = canonicalized();
}

// =========================================================================
//  Measurement -> promoted_data_type / promoted_double
// =========================================================================

Measurement Measurement::promoted_data_type(DataType target) const {
    if (data_type_ == target) return *this;

    // Only one-directional: int -> real -> complex.  String / Boolean cannot
    // promote.  Same rule as DataSeries::promoted_data_type().
    auto can_promote = [](DataType from, DataType to) -> bool {
        if (from == DataType::kInteger && to == DataType::kReal)    return true;
        if (from == DataType::kInteger && to == DataType::kComplex) return true;
        if (from == DataType::kReal    && to == DataType::kComplex) return true;
        return false;
    };
    if (!can_promote(data_type_, target))
        throw std::invalid_argument(
            "Measurement::promoted_data_type: cannot promote from " +
            std::string(DataTypeToString(data_type_)) + " to " +
            std::string(DataTypeToString(target)));

    // Unit is preserved verbatim -- mirror of DataSeries::promoted_data_type(),
    // which only copies unit_ across.  Only the dtype changes here; callers
    // that need SI values canonicalize first (see promoted_double()).
    Measurement result(*this);

    // Step up one rung at a time: int -> real -> complex.
    while (result.data_type_ != target) {
        const DataKind kind = result.shape_.kind();

        if (result.data_type_ == DataType::kInteger) {
            // int -> real
            if (kind == DataKind::kScalar) {
                result.data_type_ = DataType::kReal;
                result.storage_ =
                    static_cast<double>(boost::get<int>(result.storage_));
            } else if (kind == DataKind::kVector) {
                const VecXi& src = boost::get<VecXi>(result.storage_);
                VecXd out(src.size());
                for (Index i = 0; i < src.size(); ++i)
                    out(i) = static_cast<double>(src(i));
                result.data_type_ = DataType::kReal;
                result.storage_ = out;
            } else {
                const MatXi& src = boost::get<MatXi>(result.storage_);
                MatXd out(src.rows(), src.cols());
                for (Index r = 0; r < src.rows(); ++r)
                    for (Index c = 0; c < src.cols(); ++c)
                        out(r, c) = static_cast<double>(src(r, c));
                result.data_type_ = DataType::kReal;
                result.storage_ = out;
            }
            continue;
        }

        // real -> complex
        if (kind == DataKind::kScalar) {
            result.data_type_ = DataType::kComplex;
            result.storage_ = std::complex<double>(
                boost::get<double>(result.storage_), 0.0);
        } else if (kind == DataKind::kVector) {
            const VecXd& src = boost::get<VecXd>(result.storage_);
            VecXcd out(src.size());
            for (Index i = 0; i < src.size(); ++i)
                out(i) = std::complex<double>(src(i), 0.0);
            result.data_type_ = DataType::kComplex;
            result.storage_ = out;
        } else {
            const MatXd& src = boost::get<MatXd>(result.storage_);
            MatXcd out(src.rows(), src.cols());
            for (Index r = 0; r < src.rows(); ++r)
                for (Index c = 0; c < src.cols(); ++c)
                    out(r, c) = std::complex<double>(src(r, c), 0.0);
            result.data_type_ = DataType::kComplex;
            result.storage_ = out;
        }
    }
    return result;
}

double Measurement::promoted_double() const {
    if (shape_.kind() != DataKind::kScalar) {
        throw std::logic_error(
            "Measurement::promoted_double: not a scalar (kind=" +
            std::to_string(static_cast<int>(shape_.kind())) + ")");
    }

    const double mult = unit_.multiplier();

    switch (data_type_) {
        case DataType::kReal:
            return boost::get<double>(storage_) * mult;
        case DataType::kInteger:
            return static_cast<double>(boost::get<int>(storage_)) * mult;
        case DataType::kComplex:
            return boost::get<std::complex<double> >(storage_).real() * mult;
        default:
            throw std::invalid_argument(
                "Measurement::promoted_double: dtype is not numeric (dtype=" +
                std::to_string(static_cast<int>(data_type_)) + ")");
    }
}

// =========================================================================
// Measurement::to_dataframe
// =========================================================================

std::unique_ptr<DataFrame> Measurement::to_dataframe(
    const std::string& name) const
{
    return DataFrame::FromMeasurement(*this, name);
}

} // namespace xdataset
