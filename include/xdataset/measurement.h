#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/variant.hpp>

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "xdataset_predefine.h"
#include "units_util.h"

namespace xdataset
{

    // =========================================================================
    // Measurement -- a single named value with units (scalar | vector | matrix)
    // =========================================================================
    //
    // Measurement is the "datum" of the xdataset type system.  It is a
    // stack-friendly value type that can represent a 0-d scalar, 1-d vector
    // or 2-d matrix in any of the four supported dtypes (real, integer,
    // complex, string) and always carries a physical Unit.
    //
    // Relationship to other types:
    //   - DataSeries stores many Measurements *contiguously in memory*.
    //     Measurement is obtained via DataSeries::measurement_at(row) and serves as
    //     the user-facing value type.
    //   - DataFrame (the tabular CSV view) stores Measurement directly in
    //     DataFrameRow::fields as the cell type.
    //   - DataArray wraps a DataSeries with coordinate axes and can
    //     interact arithmetically with Measurement (future).
    //
    // Internals:
    //   - Storage is a boost::variant over all 12 (DataKind × DTypeTag)
    //     concrete types.  The variant itself lives on the stack; Eigen
    //     dynamic types allocate their element buffers on the heap, but
    //     this is expected to be modest for a single measurement.
    //   - DataKind, DTypeTag, and shape are derived from the active
    //     variant alternative at construction time and cached.
    // =========================================================================

    class Measurement
    {
    public:
        // ──────────────────────────────────
        // Storage variant -- one alternative per (DataKind, DTypeTag)
        // ──────────────────────────────────
        using Storage = boost::variant<
            // --- DataKind::kScalar ---
            double,
            int,
            std::complex<double>,
            std::string,

            // --- DataKind::kVector ---
            Eigen::VectorXd,                // DTypeTag::kReal
            Eigen::VectorXi,                // DTypeTag::kInteger
            Eigen::VectorXcd,               // DTypeTag::kComplex
            Eigen::Tensor<std::string, 1>,  // DTypeTag::kString

            // --- DataKind::kMatrix ---
            Eigen::MatrixXd,                // DTypeTag::kReal
            Eigen::MatrixXi,                // DTypeTag::kInteger
            Eigen::MatrixXcd,               // DTypeTag::kComplex
            Eigen::Tensor<std::string, 2>   // DTypeTag::kString
        >;

        // ======== construction ==============================================

        /// Default: kReal scalar 0.0, dimensionless unit.
        Measurement();

        /// Construct from a concrete value.  Kind / dtype / shape are
        /// inferred from T and (for vectors / matrices) the value's size.
        template <typename T>
        Measurement(T value);

        /// Construct from a concrete value with an explicit unit.
        template <typename T>
        Measurement(T value, Unit unit);

        // Copy / move (compiler-generated is fine -- variant is deep-copyable)
        Measurement(const Measurement&) = default;
        Measurement& operator=(const Measurement&) = default;
        Measurement(Measurement&&) = default;
        Measurement& operator=(Measurement&&) = default;

        // ======== static factories ==========================================

        /// @{
        /// Scalar factories (0-d).
        static Measurement Real(double value);
        static Measurement Integer(int value);
        static Measurement Complex(std::complex<double> value);
        static Measurement String(std::string value);
        /// @}

        /// @{
        /// Vector factories (1-d) -- numeric.
        static Measurement Vector(const Eigen::VectorXd& v);
        static Measurement Vector(const Eigen::VectorXi& v);
        static Measurement Vector(const Eigen::VectorXcd& v);
        /// @}
        static Measurement Vector(const Eigen::Tensor<std::string, 1>& v);

        /// @{
        /// Matrix factories (2-d) -- numeric.
        static Measurement Matrix(const Eigen::MatrixXd& m);
        static Measurement Matrix(const Eigen::MatrixXi& m);
        static Measurement Matrix(const Eigen::MatrixXcd& m);
        /// @}
        static Measurement Matrix(const Eigen::Tensor<std::string, 2>& m);

        // ======== metadata queries ==========================================

        DataKind kind() const { return kind_; }
        DTypeTag dtype() const { return dtype_; }
        const std::vector<Index>& shape() const { return shape_; }
        const Unit& unit() const { return unit_; }
        void set_unit(const Unit& u) { unit_ = u; }

        /// Human-readable dtype name (e.g. "Real", "Complex").
        std::string dtype_name() const;

        /// True when the stored value is not the default-constructed zero.
        bool has_value() const;

        // ======== raw storage access ========================================

        const Storage& storage() const { return storage_; }

        // ======== typed accessors ===========================================

        /// @{
        /// Scalar access -- throws std::bad_cast if T doesn't match dtype.
        template <typename T> T as_scalar() const;
        /// @}

        /// @{
        /// Vector access (returns Eigen Map for numeric, ref for string tensor).
        template <typename T>
        typename std::enable_if<
            !std::is_same<T, std::string>::value,
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>>::type
        as_vector() const;

        template <typename T>
        typename std::enable_if<
            std::is_same<T, std::string>::value,
            const Eigen::Tensor<std::string, 1>&>::type
        as_vector() const;
        /// @}

        /// @{
        /// Matrix access (returns Eigen Map for numeric, ref for string tensor).
        template <typename T>
        typename std::enable_if<
            !std::is_same<T, std::string>::value,
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>::type
        as_matrix() const;

        template <typename T>
        typename std::enable_if<
            std::is_same<T, std::string>::value,
            const Eigen::Tensor<std::string, 2>&>::type
        as_matrix() const;
        /// @}

        // ======== element access (vector / matrix -- scalar) =================

        /// Return the i-th element as a scalar Measurement (preserves unit).
        Measurement element_at(Index i) const;

        /// Return the (r, c)-th element as a scalar Measurement (preserves unit).
        Measurement element_at(Index r, Index c) const;

        // ======== formatting ================================================

        /// Return a human-readable string representation.
        std::string to_string() const;

        // ======== canonicalisation ======================================

        /// Return a canonicalised copy (value scaled to SI, unit = base_units).
        Measurement canonicalized() const;

    private:
        void infer_metadata();

        DataKind             kind_;
        DTypeTag             dtype_;
        std::vector<Index>   shape_;
        Storage              storage_;
        Unit                 unit_;
    };

    // =========================================================================
    // MeasurementFormatter -- boost::static_visitor that renders any stored
    //                         alternative to a human-readable string.
    // =========================================================================

    struct MeasurementFormatter : public boost::static_visitor<std::string>
    {
        MeasurementFormatter() = default;
        explicit MeasurementFormatter(Unit u) : unit_(std::move(u)) {}

        std::string operator()(double v) const;
        std::string operator()(int v) const;
        std::string operator()(const std::complex<double>& v) const;
        std::string operator()(const std::string& v) const;

        std::string operator()(const Eigen::VectorXd& v) const;
        std::string operator()(const Eigen::VectorXi& v) const;
        std::string operator()(const Eigen::VectorXcd& v) const;
        std::string operator()(const Eigen::Tensor<std::string, 1>& v) const;

        std::string operator()(const Eigen::MatrixXd& v) const;
        std::string operator()(const Eigen::MatrixXi& v) const;
        std::string operator()(const Eigen::MatrixXcd& v) const;
        std::string operator()(const Eigen::Tensor<std::string, 2>& v) const;

    private:
        static std::string format_complex(const std::complex<double>& v);
        static std::string format_scalar_string(const std::string& v);
        std::string with_unit(const std::string& s) const;

        Unit unit_;
    };

    // =========================================================================
    // MeasurementTypeVisitor -- extracts DataKind / DTypeTag from a variant.
    // =========================================================================

    struct MeasurementTypeVisitor : public boost::static_visitor<void>
    {
        DataKind kind   = DataKind::kScalar;
        DTypeTag dtype  = DTypeTag::kReal;

        void operator()(double)                    { kind = DataKind::kScalar;  dtype = DTypeTag::kReal;    }
        void operator()(int)                       { kind = DataKind::kScalar;  dtype = DTypeTag::kInteger; }
        void operator()(const std::complex<double>&){ kind = DataKind::kScalar;  dtype = DTypeTag::kComplex; }
        void operator()(const std::string&)         { kind = DataKind::kScalar;  dtype = DTypeTag::kString;  }

        void operator()(const Eigen::VectorXd&)             { kind = DataKind::kVector; dtype = DTypeTag::kReal;    }
        void operator()(const Eigen::VectorXi&)             { kind = DataKind::kVector; dtype = DTypeTag::kInteger; }
        void operator()(const Eigen::VectorXcd&)            { kind = DataKind::kVector; dtype = DTypeTag::kComplex; }
        void operator()(const Eigen::Tensor<std::string, 1>&){ kind = DataKind::kVector; dtype = DTypeTag::kString;  }

        void operator()(const Eigen::MatrixXd&)             { kind = DataKind::kMatrix; dtype = DTypeTag::kReal;    }
        void operator()(const Eigen::MatrixXi&)             { kind = DataKind::kMatrix; dtype = DTypeTag::kInteger; }
        void operator()(const Eigen::MatrixXcd&)            { kind = DataKind::kMatrix; dtype = DTypeTag::kComplex; }
        void operator()(const Eigen::Tensor<std::string, 2>&){ kind = DataKind::kMatrix; dtype = DTypeTag::kString;  }
    };

    // =========================================================================
    // Template implementation
    // =========================================================================

    // ── Measurement constructor ─────────────────────────────────────────────

    template <typename T>
    Measurement::Measurement(T value)
        : storage_(std::move(value))
        , unit_()
    {
        infer_metadata();
    }

    template <typename T>
    Measurement::Measurement(T value, Unit unit)
        : storage_(std::move(value))
        , unit_(std::move(unit))
    {
        infer_metadata();
    }

    // ── as_scalar<T> ────────────────────────────────────────────────────────

    template <typename T>
    T Measurement::as_scalar() const
    {
        if (kind_ != DataKind::kScalar)
            throw std::logic_error("as_scalar: Measurement is not a scalar (kind=" +
                std::to_string(static_cast<int>(kind_)) + ")");
        return boost::get<T>(storage_);
    }

    // ── as_vector<T> (numeric) ──────────────────────────────────────────────

    template <typename T>
    typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>>::type
    Measurement::as_vector() const
    {
        if (kind_ != DataKind::kVector)
            throw std::logic_error("as_vector: Measurement is not a vector (kind=" +
                std::to_string(static_cast<int>(kind_)) + ")");
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VecType;
        const VecType& vec = boost::get<VecType>(storage_);
        return Eigen::Map<const VecType>(vec.data(), vec.size());
    }

    // ── as_vector<T> (string) ───────────────────────────────────────────────

    template <typename T>
    typename std::enable_if<
        std::is_same<T, std::string>::value,
        const Eigen::Tensor<std::string, 1>&>::type
    Measurement::as_vector() const
    {
        if (kind_ != DataKind::kVector)
            throw std::logic_error("as_vector: Measurement is not a vector (kind=" +
                std::to_string(static_cast<int>(kind_)) + ")");
        return boost::get<Eigen::Tensor<std::string, 1>>(storage_);
    }

    // ── as_matrix<T> (numeric) ──────────────────────────────────────────────

    template <typename T>
    typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>::type
    Measurement::as_matrix() const
    {
        if (kind_ != DataKind::kMatrix)
            throw std::logic_error("as_matrix: Measurement is not a matrix (kind=" +
                std::to_string(static_cast<int>(kind_)) + ")");
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatType;
        const MatType& mat = boost::get<MatType>(storage_);
        return Eigen::Map<const MatType>(mat.data(), mat.rows(), mat.cols());
    }

    // ── as_matrix<T> (string) ───────────────────────────────────────────────

    template <typename T>
    typename std::enable_if<
        std::is_same<T, std::string>::value,
        const Eigen::Tensor<std::string, 2>&>::type
    Measurement::as_matrix() const
    {
        if (kind_ != DataKind::kMatrix)
            throw std::logic_error("as_matrix: Measurement is not a matrix (kind=" +
                std::to_string(static_cast<int>(kind_)) + ")");
        return boost::get<Eigen::Tensor<std::string, 2>>(storage_);
    }

    // =========================================================================
    //  Measurement arithmetic operators
    // =========================================================================

    /// DataKind promotion for binary ops (per the Scalar/Vector/Matrix table).
    /// Returns the result kind, or throws on incompatible combinations.
    DataKind promoted_kind(DataKind a, DataKind b);

    /// dtype promotion: Integer ⊂ Real ⊂ Complex.  String throws.
    DTypeTag promoted_dtype(DTypeTag a, DTypeTag b);

    // Measurement – Measurement
    Measurement operator+(const Measurement& lhs, const Measurement& rhs);
    Measurement operator-(const Measurement& lhs, const Measurement& rhs);
    Measurement operator*(const Measurement& lhs, const Measurement& rhs);
    Measurement operator/(const Measurement& lhs, const Measurement& rhs);

    /// pow(base, exponent): exponent must be dimensionless, non-String.
    /// DataKind broadcasting applies (e.g. Scalar^Vector → Vector).
    /// When exponent is non-scalar, base must also be dimensionless.
    Measurement pow(const Measurement& base, const Measurement& exponent);

} // namespace xdataset

#endif // MEASUREMENT_H
