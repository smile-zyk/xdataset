#include "measurement.h"
#include "unit.h"

#include <sstream>
#include <stdexcept>

namespace xdataset
{

    // =========================================================================
    // Measurement -- metadata inference
    // =========================================================================

    void Measurement::infer_metadata()
    {
        MeasurementTypeVisitor v;
        boost::apply_visitor(v, storage_);
        kind_  = v.kind;
        dtype_ = v.dtype;

        // Derive shape by dispatching on (kind, dtype).
        shape_.clear();
        switch (kind_)
        {
            case DataKind::kScalar:
                break;  // shape_ remains empty

            case DataKind::kVector:
                switch (dtype_)
                {
                    case DTypeTag::kReal:    shape_.push_back(boost::get<Eigen::VectorXd>(storage_).size());            break;
                    case DTypeTag::kInteger: shape_.push_back(boost::get<Eigen::VectorXi>(storage_).size());            break;
                    case DTypeTag::kComplex: shape_.push_back(boost::get<Eigen::VectorXcd>(storage_).size());           break;
                    case DTypeTag::kString:  shape_.push_back(boost::get<Eigen::Tensor<std::string, 1>>(storage_).dimension(0)); break;
                }
                break;

            case DataKind::kMatrix:
                switch (dtype_)
                {
                    case DTypeTag::kReal:
                    {
                        const auto& m = boost::get<Eigen::MatrixXd>(storage_);
                        shape_.push_back(m.rows()); shape_.push_back(m.cols());
                        break;
                    }
                    case DTypeTag::kInteger:
                    {
                        const auto& m = boost::get<Eigen::MatrixXi>(storage_);
                        shape_.push_back(m.rows()); shape_.push_back(m.cols());
                        break;
                    }
                    case DTypeTag::kComplex:
                    {
                        const auto& m = boost::get<Eigen::MatrixXcd>(storage_);
                        shape_.push_back(m.rows()); shape_.push_back(m.cols());
                        break;
                    }
                    case DTypeTag::kString:
                    {
                        const auto& t = boost::get<Eigen::Tensor<std::string, 2>>(storage_);
                        shape_.push_back(t.dimension(0)); shape_.push_back(t.dimension(1));
                        break;
                    }
                }
                break;
        }
    }

    // =========================================================================
    // Measurement -- default ctor
    // =========================================================================

    Measurement::Measurement()
        : kind_(DataKind::kScalar)
        , dtype_(DTypeTag::kReal)
        , shape_()
        , storage_(0.0)
        , unit_()
    {
    }

    // =========================================================================
    // Measurement -- static factories
    // =========================================================================

    Measurement Measurement::Real(double value)
    {
        Measurement m;
        m.storage_ = value;
        m.kind_    = DataKind::kScalar;
        m.dtype_   = DTypeTag::kReal;
        return m;
    }

    Measurement Measurement::Integer(int value)
    {
        Measurement m;
        m.storage_ = value;
        m.kind_    = DataKind::kScalar;
        m.dtype_   = DTypeTag::kInteger;
        return m;
    }

    Measurement Measurement::Complex(std::complex<double> value)
    {
        Measurement m;
        m.storage_ = value;
        m.kind_    = DataKind::kScalar;
        m.dtype_   = DTypeTag::kComplex;
        return m;
    }

    Measurement Measurement::String(std::string value)
    {
        Measurement m;
        m.storage_ = value;
        m.kind_    = DataKind::kScalar;
        m.dtype_   = DTypeTag::kString;
        return m;
    }

    Measurement Measurement::Vector(const Eigen::VectorXd& v)
    {
        Measurement m;
        m.storage_ = v;
        m.kind_    = DataKind::kVector;
        m.dtype_   = DTypeTag::kReal;
        m.shape_.push_back(v.size());
        return m;
    }

    Measurement Measurement::Vector(const Eigen::VectorXi& v)
    {
        Measurement m;
        m.storage_ = v;
        m.kind_    = DataKind::kVector;
        m.dtype_   = DTypeTag::kInteger;
        m.shape_.push_back(v.size());
        return m;
    }

    Measurement Measurement::Vector(const Eigen::VectorXcd& v)
    {
        Measurement m;
        m.storage_ = v;
        m.kind_    = DataKind::kVector;
        m.dtype_   = DTypeTag::kComplex;
        m.shape_.push_back(v.size());
        return m;
    }

    Measurement Measurement::Vector(const Eigen::Tensor<std::string, 1>& v)
    {
        Measurement m;
        m.storage_ = v;
        m.kind_    = DataKind::kVector;
        m.dtype_   = DTypeTag::kString;
        m.shape_.push_back(v.dimension(0));
        return m;
    }

    Measurement Measurement::Matrix(const Eigen::MatrixXd& m)
    {
        Measurement mm;
        mm.storage_ = m;
        mm.kind_    = DataKind::kMatrix;
        mm.dtype_   = DTypeTag::kReal;
        mm.shape_.push_back(m.rows());
        mm.shape_.push_back(m.cols());
        return mm;
    }

    Measurement Measurement::Matrix(const Eigen::MatrixXi& m)
    {
        Measurement mm;
        mm.storage_ = m;
        mm.kind_    = DataKind::kMatrix;
        mm.dtype_   = DTypeTag::kInteger;
        mm.shape_.push_back(m.rows());
        mm.shape_.push_back(m.cols());
        return mm;
    }

    Measurement Measurement::Matrix(const Eigen::MatrixXcd& m)
    {
        Measurement mm;
        mm.storage_ = m;
        mm.kind_    = DataKind::kMatrix;
        mm.dtype_   = DTypeTag::kComplex;
        mm.shape_.push_back(m.rows());
        mm.shape_.push_back(m.cols());
        return mm;
    }

    Measurement Measurement::Matrix(const Eigen::Tensor<std::string, 2>& m)
    {
        Measurement mm;
        mm.storage_ = m;
        mm.kind_    = DataKind::kMatrix;
        mm.dtype_   = DTypeTag::kString;
        mm.shape_.push_back(m.dimension(0));
        mm.shape_.push_back(m.dimension(1));
        return mm;
    }

    // =========================================================================
    // Measurement ?queries
    // =========================================================================

    std::string Measurement::dtype_name() const
    {
        switch (dtype_)
        {
            case DTypeTag::kReal:    return "Real";
            case DTypeTag::kInteger: return "Integer";
            case DTypeTag::kComplex: return "Complex";
            case DTypeTag::kString:  return "String";
        }
        return "Unknown";
    }

    bool Measurement::has_value() const
    {
        // Default-constructed Measurement is 0.0 real scalar -- always has a value.
        // (variant is never empty.)
        return true;
    }

    std::string Measurement::to_string() const
    {
        MeasurementFormatter fmt(unit_);
        return boost::apply_visitor(fmt, storage_);
    }

    // =========================================================================
    // Measurement -- element_at (vector ?scalar / matrix ?scalar)
    // =========================================================================

    Measurement Measurement::element_at(Index i) const
    {
        if (kind_ != DataKind::kVector)
            throw std::logic_error("element_at(Index) requires vector data");
        switch (dtype_)
        {
            case DTypeTag::kReal:    return Measurement(boost::get<Eigen::VectorXd>(storage_)(i), unit_);
            case DTypeTag::kInteger: return Measurement(boost::get<Eigen::VectorXi>(storage_)(i), unit_);
            case DTypeTag::kComplex: return Measurement(boost::get<Eigen::VectorXcd>(storage_)(i), unit_);
            case DTypeTag::kString:  return Measurement(boost::get<Eigen::Tensor<std::string, 1>>(storage_)(i), unit_);
        }
        throw std::logic_error("unsupported dtype");
    }

    Measurement Measurement::element_at(Index r, Index c) const
    {
        if (kind_ != DataKind::kMatrix)
            throw std::logic_error("element_at(Index, Index) requires matrix data");
        switch (dtype_)
        {
            case DTypeTag::kReal:    return Measurement(boost::get<Eigen::MatrixXd>(storage_)(r, c), unit_);
            case DTypeTag::kInteger: return Measurement(boost::get<Eigen::MatrixXi>(storage_)(r, c), unit_);
            case DTypeTag::kComplex: return Measurement(boost::get<Eigen::MatrixXcd>(storage_)(r, c), unit_);
            case DTypeTag::kString:  return Measurement(boost::get<Eigen::Tensor<std::string, 2>>(storage_)(r, c), unit_);
        }
        throw std::logic_error("unsupported dtype");
    }

    // =========================================================================
    // MeasurementFormatter
    // =========================================================================

    std::string MeasurementFormatter::with_unit(const std::string& s) const
    {
        if (!unit_.has_dimension())
            return s;
        return s + " " + unit_.to_string();
    }

    // --- scalar with auto-scale ------------------------------------------

    std::string MeasurementFormatter::operator()(double v) const
    {
        auto bd = unit_.best_display(v);
        std::ostringstream oss;
        oss << (v * bd.scale);
        if (bd.name.empty())
            return oss.str();
        return oss.str() + " " + bd.name;
    }

    std::string MeasurementFormatter::operator()(int v) const
    {
        auto bd = unit_.best_display(static_cast<double>(v));
        std::ostringstream oss;
        oss << (v * bd.scale);
        if (bd.name.empty())
            return oss.str();
        return oss.str() + " " + bd.name;
    }

    std::string MeasurementFormatter::operator()(const std::complex<double>& v) const
    {
        auto bd = unit_.best_display(v.real());
        std::ostringstream oss;
        oss << (v.real() * bd.scale);
        double imag = v.imag() * bd.scale;
        if (imag >= 0.0) oss << "+";
        oss << imag << "i";
        if (bd.name.empty())
            return oss.str();
        return oss.str() + " " + bd.name;
    }

    std::string MeasurementFormatter::operator()(const std::string& v) const
    {
        return v;
    }

    // ── vector ─────────────────────────────────────────────────────────────

    std::string MeasurementFormatter::operator()(const Eigen::VectorXd& v) const
    {
        std::ostringstream oss;
        oss << "[";
        for (Index i = 0; i < v.size(); ++i)
        {
            if (i > 0) oss << ",";
            oss << v(i);
        }
        oss << "]";
        return with_unit(oss.str());
    }

    std::string MeasurementFormatter::operator()(const Eigen::VectorXi& v) const
    {
        std::ostringstream oss;
        oss << "[";
        for (Index i = 0; i < v.size(); ++i)
        {
            if (i > 0) oss << ",";
            oss << v(i);
        }
        oss << "]";
        return with_unit(oss.str());
    }

    std::string MeasurementFormatter::operator()(const Eigen::VectorXcd& v) const
    {
        auto bd = unit_.best_display(v.size() > 0 ? v(0).real() : 0.0);
        std::ostringstream oss;
        oss << "[";
        for (Index i = 0; i < v.size(); ++i)
        {
            if (i > 0) oss << ",";
            double r = v(i).real() * bd.scale;
            double im = v(i).imag() * bd.scale;
            oss << r;
            if (im >= 0.0) oss << "+";
            oss << im << "i";
        }
        oss << "]";
        if (bd.name.empty())
            return oss.str();
        return oss.str() + " " + bd.name;
    }

    std::string MeasurementFormatter::operator()(const Eigen::Tensor<std::string, 1>& v) const
    {
        std::ostringstream oss;
        oss << "[";
        for (Index i = 0; i < v.dimension(0); ++i)
        {
            if (i > 0) oss << ",";
            oss << v(i);
        }
        oss << "]";
        return oss.str();
    }

    // ── matrix ─────────────────────────────────────────────────────────────

    std::string MeasurementFormatter::operator()(const Eigen::MatrixXd& v) const
    {
        std::ostringstream oss;
        oss << "[";
        for (Index r = 0; r < v.rows(); ++r)
        {
            if (r > 0) oss << ",";
            oss << "[";
            for (Index c = 0; c < v.cols(); ++c)
            {
                if (c > 0) oss << ",";
                oss << v(r, c);
            }
            oss << "]";
        }
        oss << "]";
        return with_unit(oss.str());
    }

    std::string MeasurementFormatter::operator()(const Eigen::MatrixXi& v) const
    {
        std::ostringstream oss;
        oss << "[";
        for (Index r = 0; r < v.rows(); ++r)
        {
            if (r > 0) oss << ",";
            oss << "[";
            for (Index c = 0; c < v.cols(); ++c)
            {
                if (c > 0) oss << ",";
                oss << v(r, c);
            }
            oss << "]";
        }
        oss << "]";
        return with_unit(oss.str());
    }

    std::string MeasurementFormatter::operator()(const Eigen::MatrixXcd& v) const
    {
        auto bd = unit_.best_display(v.size() > 0 ? v(0, 0).real() : 0.0);
        std::ostringstream oss;
        oss << "[";
        for (Index r = 0; r < v.rows(); ++r)
        {
            if (r > 0) oss << ",";
            oss << "[";
            for (Index c = 0; c < v.cols(); ++c)
            {
                if (c > 0) oss << ",";
                double rv = v(r, c).real() * bd.scale;
                double im = v(r, c).imag() * bd.scale;
                oss << rv;
                if (im >= 0.0) oss << "+";
                oss << im << "i";
            }
            oss << "]";
        }
        oss << "]";
        if (bd.name.empty())
            return oss.str();
        return oss.str() + " " + bd.name;
    }

    std::string MeasurementFormatter::operator()(const Eigen::Tensor<std::string, 2>& v) const
    {
        std::ostringstream oss;
        oss << "[";
        for (Index r = 0; r < v.dimension(0); ++r)
        {
            if (r > 0) oss << ",";
            oss << "[";
            for (Index c = 0; c < v.dimension(1); ++c)
            {
                if (c > 0) oss << ",";
                oss << v(r, c);
            }
            oss << "]";
        }
        oss << "]";
        return oss.str();
    }

// =========================================================================
//  Measurement — canonicalized
// =========================================================================

Measurement Measurement::canonicalized() const {
    if (dtype_ == DTypeTag::kString) {
        Measurement result(*this);
        result.unit_ = unit_.canonicalized();
        return result;
    }

    double mult = unit_.multiplier();
    Unit target = unit_.canonicalized();
    bool affine = unit_.is_affine();

    // Fast path: already canonical
    if (!affine && mult == 1.0) {
        Measurement result(*this);
        result.unit_ = target;
        return result;
    }

    // Promote integer to real when scaling is needed
    DTypeTag res_dtype = (dtype_ == DTypeTag::kInteger) ? DTypeTag::kReal : dtype_;

    Measurement result;
    result.kind_  = kind_;
    result.dtype_ = res_dtype;
    result.shape_ = shape_;
    result.unit_  = target;

    Index count = 1;
    if (kind_ == DataKind::kVector) count = shape_[0];
    else if (kind_ == DataKind::kMatrix) count = shape_[0] * shape_[1];

    if (kind_ == DataKind::kScalar) {
        double v = (dtype_ == DTypeTag::kInteger)
                   ? static_cast<double>(boost::get<int>(storage_))
                   : boost::get<double>(storage_);
        v = affine ? units::convert(v, unit_.raw(), target.raw()) : v * mult;
        if (res_dtype == DTypeTag::kComplex) {
            std::complex<double> cv = boost::get<std::complex<double> >(storage_);
            if (!affine) cv *= mult;
            result.storage_ = cv;
        } else {
            result.storage_ = v;
        }
        return result;
    }

    if (kind_ == DataKind::kVector) {
        if (res_dtype == DTypeTag::kComplex) {
            Eigen::VectorXcd vec = boost::get<Eigen::VectorXcd>(storage_);
            if (!affine) vec *= mult;
            result.storage_ = vec;
        } else {
            Eigen::VectorXd vec(count);
            for (Index i = 0; i < count; ++i) {
                double v = (dtype_ == DTypeTag::kInteger)
                           ? static_cast<double>(boost::get<Eigen::VectorXi>(storage_)(i))
                           : boost::get<Eigen::VectorXd>(storage_)(i);
                vec(i) = affine ? units::convert(v, unit_.raw(), target.raw()) : v * mult;
            }
            result.storage_ = vec;
        }
        return result;
    }

    // Matrix
    if (res_dtype == DTypeTag::kComplex) {
        Eigen::MatrixXcd mat = boost::get<Eigen::MatrixXcd>(storage_);
        if (!affine) mat *= mult;
        result.storage_ = mat;
    } else {
        Index rows = shape_[0], cols = shape_[1];
        Eigen::MatrixXd mat(rows, cols);
        for (Index r = 0; r < rows; ++r) {
            for (Index c = 0; c < cols; ++c) {
                double v = (dtype_ == DTypeTag::kInteger)
                           ? static_cast<double>(boost::get<Eigen::MatrixXi>(storage_)(r, c))
                           : boost::get<Eigen::MatrixXd>(storage_)(r, c);
                mat(r, c) = affine ? units::convert(v, unit_.raw(), target.raw()) : v * mult;
            }
        }
        result.storage_ = mat;
    }
    return result;
}

bool Measurement::is_canonicalized() const {
    return unit_.is_canonical();
}

} // namespace xdataset
