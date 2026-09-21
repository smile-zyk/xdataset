#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/variant.hpp>

#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "xdataset_predefine.h"
#include "data_shape.h"
#include "unit.h"
#include "multi_index_selector.h"

namespace xdataset
{

    class DataFrame;
    class Measurement;

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
    //   - Storage is a boost::variant over all 12 (DataKind x DataType)
    //     concrete types.  The variant itself lives on the stack; Eigen
    //     dynamic types allocate their element buffers on the heap, but
    //     this is expected to be modest for a single measurement.
    //   - DataKind, DataType, and shape are derived from the active
    //     variant alternative at construction time and cached.
    // =========================================================================

    // =========================================================================
    //  Display format configuration
    // =========================================================================

    /// How a real / integer number is rendered.
    enum class NumberFormat
    {
        /// All digits before the decimal point are shown; no exponent and NO
        /// unit auto-scaling (the value stays in its own unit).
        /// 1530000.123 Hz with 6 significant digits -> "1530000 Hz"
        ///
        /// This is the DEFAULT mode.
        kFull,
        /// Scientific notation, no unit auto-scaling.
        /// 1000 Hz with 3 significant digits -> "1.00e3 Hz"
        kScientific,
        /// Engineering notation: the exponent is forced to a multiple of 3
        /// and is expressed as an SI prefix on the unit.
        /// 1000 Hz -> "1 kHz".
        kEngineering,
        /// Hexadecimal (base 16), "0x" prefixed.
        kHex,
        /// Octal (base 8), "0" prefixed.
        kOctal,
        /// Binary (base 2), "0b" prefixed.
        kBinary
    };

    /// How a complex number is rendered.
    ///
    /// Only kRealImaginary carries a unit (and therefore a scale).  Every
    /// other mode is a pure "a/b" pair -- magnitude (or dB) and phase -- with
    /// NO unit and NO scaling, because a magnitude/phase pair is not a value
    /// in the Measurement's unit the way a real part is.
    enum class ComplexFormat
    {
        /// Real / imaginary, with the unit:  "778-258i mV"
        kRealImaginary,
        /// Magnitude / phase in degrees:  "0.819663/-18.3465"
        kMagDegrees,
        /// dB magnitude / phase in degrees:  "-1.72729/-18.3465"
        kDbDegrees,
        /// Magnitude / phase in radians:  "0.819663/-0.320207"
        kMagRadians,
        /// dB magnitude / phase in radians:  "-1.72729/-0.320207"
        kDbRadians
    };

    /// Display options applied to every Measurement rendered with them.
    ///
    /// A plain 16-byte value type: passed by const reference and never
    /// mutated.  Rendering is a pure function of (value, unit, options), so
    /// there is no formatter object and no per-call state to manage.
    struct FormatOptions
    {
        NumberFormat  number_format   = NumberFormat::kFull;
        ComplexFormat complex_format  = ComplexFormat::kRealImaginary;

        /// Significant digits used by kFull / kScientific / kEngineering.
        /// Ignored by kHex / kOctal / kBinary (those are exact integer
        /// representations).  Clamped to [1, 17].
        int significant_digits = 6;

        /// Whether the unit suffix is appended.  The unit itself is the
        /// Measurement's own (see Measurement::unit()) -- there is nothing to
        /// configure about WHICH unit, only whether to show it.
        ///
        /// When false the value is rendered bare and is NOT auto-scaled:
        /// the scale is expressed through the unit prefix, so hiding the unit
        /// means hiding the scale too (0.002 V -> "0.002", not "2").
        ///
        /// Note: scale prefixes (m / M / K / G ...) are not units, but they
        /// are part of the unit suffix and are shown whenever this is true
        /// and the mode auto-scales (kEngineering) -- that is what "best unit
        /// display" means (1000 Hz -> "1 kHz").
        ///
        /// When false the value is rendered bare and is NOT auto-scaled --
        /// EXCEPT in kEngineering, where the 10^3 step is part of the mode
        /// itself, so the prefix is kept and appended SPICE-style with no
        /// space: 2.4e9 Hz -> "2.4G".
        bool show_unit = true;
    };

    // =========================================================================
    //  Formatting API
    // =========================================================================
    //
    //  Rendering is a PURE FUNCTION: (value, unit, options) -> string.  There
    //  is no formatter object, no visitor, and no per-call mutable state, so
    //  every function below is re-entrant and thread-safe.
    //
    //  Three ways to supply the options, all equally cheap (options are
    //  always passed by const reference -- 16 bytes, never copied):
    //
    //    1. Global default -- set once, used by to_string():
    //         FormatDefaults::Instance().Set(o);
    //         m.to_string();
    //
    //    2. One-off -- pass them explicitly:
    //         m.to_string(o);
    //
    //    3. Scoped override -- RAII, restores on exit (also on exception):
    //         { FormatScope scope(o);  render_whole_table();  }
    //
    //  Hosts that render many values sharing one unit (e.g. a table column)
    //  can hoist the expensive part out of the loop:
    //         DisplayScale s = ResolveScale(magnitude, unit, opts);  // once
    //         FormatWithScale(v, s, opts.number_format, opts.significant_digits);
    // =========================================================================

    /// How a value must be scaled and which suffix (if any) to append.
    ///
    /// Produced by ResolveScale() -- the only expensive step of rendering
    /// (it may reach Unit::best_display / UnitRegistry::decompose).
    ///
    /// The suffix is PRE-FORMATTED: it already carries the leading space for
    /// a unit (" GHz") or none for a SPICE prefix ("G"), so the caller just
    /// appends it.
    struct DisplayScale
    {
        double      scale = 1.0;   ///< display_value = raw_value * scale
        std::string suffix;        ///< " GHz", "G" (SPICE), or ""
    };

    /// Resolve the display scale + suffix for a value of magnitude @p v.
    /// Call once per unit (or per column), then reuse via FormatWithScale().
    XDATASET_API DisplayScale ResolveScale(double v, const Unit& unit,
                                           const FormatOptions& options);

    /// Render one number using an ALREADY-RESOLVED scale.  Cheap: no unit
    /// lookup, just the number formatting plus the suffix.
    XDATASET_API std::string FormatWithScale(double v, const DisplayScale& scale,
                                             NumberFormat format, int digits);

    /// Render one integer using an already-resolved scale.  An integer never
    /// gets a decimal point in kFull mode.
    XDATASET_API std::string FormatWithScale(int v, const DisplayScale& scale,
                                             NumberFormat format, int digits);

    /// Render a whole Measurement.  This is the general entry point.
    XDATASET_API std::string Format(const Measurement& m,
                                    const FormatOptions& options);

    /// The process-wide default options.
    ///
    /// Measurement::to_string() (no-argument) renders with these.  Set them
    /// once at start-up or when the user changes a setting; the hot path only
    /// reads them.
    ///
    /// NOT thread_local: the options are read-only during rendering, and the
    /// unit is a function parameter, so there is no per-call mutable state.
    class XDATASET_API FormatDefaults
    {
    public:
        static FormatDefaults& Instance();

        /// Replace the default options (copied in; @p options need not
        /// outlive this call).
        void Set(const FormatOptions& options);

        const FormatOptions& Get() const { return options_; }

    private:
        FormatDefaults() = default;
        FormatOptions options_;
    };

    /// RAII override of the default options for the current scope.
    ///
    /// Restores the previous options on destruction, so it is exception-safe
    /// and nestable -- unlike a manual save/restore around a call.
    class XDATASET_API FormatScope
    {
    public:
        explicit FormatScope(const FormatOptions& options);
        ~FormatScope();

        FormatScope(const FormatScope&) = delete;
        FormatScope& operator=(const FormatScope&) = delete;

    private:
        FormatOptions saved_;
    };

    class XDATASET_API Measurement
    {
    public:
        // ----------------------------------
        // Storage variant -- one alternative per (DataKind, DataType)
        // ----------------------------------
        using Storage = boost::variant<
            // --- DataKind::kScalar ---
            double,
            int,
            std::complex<double>,
            std::string,
            bool,                           // DataType::kBoolean

            // --- DataKind::kVector ---
            VecXd,                          // DataType::kReal     (1 行, w 列, RowMajor)
            VecXi,                          // DataType::kInteger
            VecXcd,                         // DataType::kComplex
            VecXs,                          // DataType::kString

            // --- DataKind::kMatrix ---
            MatXd,                          // DataType::kReal    (RowMajor)
            MatXi,                          // DataType::kInteger
            MatXcd,                         // DataType::kComplex
            MatXs                           // DataType::kString
        >;

        // ======== construction ==============================================

        /// Default: kReal scalar 0.0, dimensionless unit.
        Measurement();

        // Copy / move (compiler-generated is fine -- variant is deep-copyable)
        Measurement(const Measurement&) = default;
        Measurement& operator=(const Measurement&) = default;
        Measurement(Measurement&&) = default;
        Measurement& operator=(Measurement&&) = default;

        // ======== static factories ==========================================

        /// @{
        /// Scalar factories (0-d).  Real / Integer / Complex carry an optional
        /// unit (default: dimensionless).  Boolean and String values cannot
        /// carry a physical unit, so their factories take no unit.
        static Measurement Real(double value, const Unit& u = Unit());
        static Measurement Integer(int value, const Unit& u = Unit());
        static Measurement Complex(std::complex<double> value, const Unit& u = Unit());
        static Measurement String(std::string value);
        static Measurement Boolean(bool value);
        /// @}

        /// @{
        /// Vector factories (1-d) -- numeric (行向量), with optional unit.
        /// By-value parameters: lvalues copy, rvalues move (no extra copy for
        /// `Measurement::Vector(MatXi(...))` style temporaries).
        static Measurement Vector(VecXd v, const Unit& u = Unit());
        static Measurement Vector(VecXi v, const Unit& u = Unit());
        static Measurement Vector(VecXcd v, const Unit& u = Unit());
        /// @}
        /// Vector factory from an Eigen Map view (e.g. DataSeries::vector_at).
        /// Copies the viewed data into a standalone Measurement.
        static Measurement Vector(VecConstMap<double> v, const Unit& u = Unit());
        static Measurement Vector(VecConstMap<int> v, const Unit& u = Unit());
        static Measurement Vector(VecConstMap<std::complex<double> > v,
                                  const Unit& u = Unit());
        static Measurement Vector(const VecXs& v);   // string rows: no unit

        /// @{
        /// Scalar factory dispatching on the element type T.  Used internally
        /// by transform() when the callback returns a scalar of an arbitrary
        /// type; the appropriate Real / Integer / Complex factory is selected
        /// at compile time.  String outputs carry no unit.
        template <typename T>
        static Measurement Scalar(const T& v, const Unit& u = Unit());
        /// @}

        /// @{
        /// Matrix factories (2-d) -- numeric (RowMajor), with optional unit.
        /// By-value parameters: lvalues copy, rvalues move.
        static Measurement Matrix(MatXd m, const Unit& u = Unit());
        static Measurement Matrix(MatXi m, const Unit& u = Unit());
        static Measurement Matrix(MatXcd m, const Unit& u = Unit());
        /// @}
        /// Matrix factory from an Eigen Map view (e.g. DataSeries::matrix_at).
        /// Copies the viewed data into a standalone Measurement.
        static Measurement Matrix(MatConstMap<double> m, const Unit& u = Unit());
        static Measurement Matrix(MatConstMap<int> m, const Unit& u = Unit());
        static Measurement Matrix(MatConstMap<std::complex<double> > m,
                                  const Unit& u = Unit());
        static Measurement Matrix(const MatXs& m);   // string cells: no unit

        // ======== metadata queries ==========================================

        DataKind data_kind() const { return shape_.kind(); }
        DataType data_type() const { return data_type_; }
        const DataShape& shape() const { return shape_; }
        const Unit& unit() const { return unit_; }

        /// Number of elements in one cell: scalar=1, vector=shape[0], matrix=shape[0]*shape[1]
        Index element_count() const { return shape_.element_count(); }

        Measurement& set_unit(const Unit& u) {
            if (data_type_ == DataType::kBoolean) {
                if (u.has_dimension())
                    throw std::invalid_argument("Boolean measurements cannot have a unit");
            }
            unit_ = u;
            return *this;
        }

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
            VecConstMap<T>>::type
        as_vector() const;

        template <typename T>
        typename std::enable_if<
            std::is_same<T, std::string>::value,
            const VecXs&>::type
        as_vector() const;
        /// @}

        /// @{
        /// Matrix access (returns Eigen Map for numeric, ref for string tensor).
        template <typename T>
        typename std::enable_if<
            !std::is_same<T, std::string>::value,
            MatConstMap<T>>::type
        as_matrix() const;

        template <typename T>
        typename std::enable_if<
            std::is_same<T, std::string>::value,
            const MatXs&>::type
        as_matrix() const;
        /// @}

        // ======== element access (vector / matrix -- scalar) =================

        /// Return the i-th element as a scalar Measurement (preserves unit).
        Measurement element_at(Index i) const;

        /// Return the (r, c)-th element as a scalar Measurement (preserves unit).
        Measurement element_at(Index r, Index c) const;

        /// Return a sub-Measurement selected by MultiIndexSelectors.
        /// For vectors: 1 selector → scalar (if single) or sub-vector.
        /// For matrices: 2 selectors → scalar/vector/sub-matrix.
        /// Preserves unit.  Not valid for scalar data.
        Measurement at(const std::vector<MultiIndexSelector>& selectors) const;

        // ======== per-element transform ====================================

        /// Apply a function to each scalar element of this Measurement.
        /// Shape, kind, and unit are preserved.  The output dtype is deduced
        /// from the return type of `func`, so it may differ from the input
        /// dtype (e.g. complex → real for abs, double → int for round).
        ///
        /// For Scalars: func is called once on the scalar value.
        /// For Vectors: func is called on each element of the vector.
        /// For Matrices: func is called on each element of the matrix.
        ///
        /// The function must be callable with the Measurement's scalar type
        /// (double, int, std::complex<double>, or std::string).  Passing a
        /// function whose argument type does not match the Measurement's
        /// data type is a compile error.
        ///
        /// Example:
        ///   auto v = Value::Vector(VecXd{1.0, 2.0, 3.0});
        ///   Measurement sq = v.as_measurement().transform([](double x) { return x * x; });
        template <typename Func>
        Measurement transform(Func&& func) const
        {
            switch (shape_.kind()) {
                case DataKind::kScalar:
                    return transform_scalar_dispatch(std::forward<Func>(func));
                case DataKind::kVector:
                    return transform_vector_dispatch(std::forward<Func>(func));
                case DataKind::kMatrix:
                    return transform_matrix_dispatch(std::forward<Func>(func));
            }
            return *this;
        }

        // ======== formatting ================================================

        /// Return a human-readable string representation using the default
        /// format options (see FormatDefaults).
        std::string to_string() const;

        /// Return a human-readable string representation using @p options.
        std::string to_string(const FormatOptions& options) const;

        // ======== equality ================================================

        /// Compare two Measurements for value equality.
        ///
        /// Two Measurements are equal iff they have the same data type, the
        /// same shape, and equal underlying values (compared element-wise,
        /// since Eigen matrices and string tensors do not define operator==).
        /// The unit is NOT compared: a value is equal to the same value with
        /// any unit.  Use unit().same_dimension() to compare units.
        ///
        /// Coordinates (scalar Measurements) are the primary use case, but
        /// vector/matrix Measurements are also handled element-wise.
        bool operator==(const Measurement& other) const;

        bool operator!=(const Measurement& other) const { return !(*this == other); }

        // ======== DataFrame conversion ======================================

        /// Create a single-row DataFrame with this measurement as the only
        /// row, using \p name as the column header prefix.
        /// The concrete frame type is internal to the library, hence the
        /// opaque unique_ptr<DataFrame> return.
        std::unique_ptr<DataFrame> to_dataframe(const std::string& name) const;

        // ======== unit conversion ======================================

        /// Convert to \p target unit and return the result.  This is a pure
        /// unit-domain operation: the shape (scalar / vector / matrix) is
        /// preserved and every element is scaled by
        ///     factor = unit().multiplier() / target.multiplier()
        /// (SI value = stored value * multiplier, so conversion is the ratio
        /// of the two multipliers).
        ///
        ///   500 mV -> V   : 0.5
        ///   0.5 V  -> mV  : 500
        ///   2 kHz  -> Hz  : 2000
        ///
        /// dtype handling:
        ///   - kReal / kComplex : scaled in place (complex: both parts).
        ///   - kInteger         : kept as int when factor == 1, otherwise
        ///                        promoted to kReal (int * 1e-3 truncates).
        ///   - kString / kBoolean: no numeric value to scale; only the unit
        ///                        tag is replaced.
        ///
        /// Throws std::invalid_argument when both units carry a dimension and
        /// they differ (unit().same_dimension(target) == false).
        ///
        /// canonicalized() is exactly converted_to(unit().canonicalized()).
        Measurement converted_to(const Unit& target) const;

        // ======== canonicalisation ======================================

        /// Convert in-place to canonical SI (value scaled, unit = base_units).
        /// Equivalent to `*this = canonicalized()`.
        void canonicalize();

        /// Return a canonicalised copy (value scaled to SI, unit = base_units).
        Measurement canonicalized() const;

        /// True when the stored unit is already canonical (multiplier == 1, non-affine).
        bool is_canonicalized() const;

        // ======== dtype promotion ========================================

        /// Promote the dtype: int -> real -> complex.  No-op when already at
        /// or above \p target.  String / Boolean are not promotable.
        /// Shape and unit are preserved verbatim; returns a new Measurement.
        ///
        /// Mirror of DataSeries::promoted_data_type() -- same one-directional
        /// rule, same exception on an impossible promotion, and the same
        /// "unit is copied, not converted" behaviour.  This is a pure dtype
        /// operation: it never touches values or units.
        ///
        /// Throws std::invalid_argument when the promotion is not possible
        /// (including any narrowing, e.g. complex -> real).
        Measurement promoted_data_type(DataType target) const;

        // ======== numeric conversion ======================================

        /// Canonicalised scalar narrowed to a plain \c double -- the unit
        /// multiplier is absorbed and the value is read as kReal:
        ///   500 mV -> 0.5,   1.5 V -> 1.5,   2 kHz -> 2000.
        ///
        /// Why this exists alongside promoted_data_type(): the promotion
        /// ladder is one-directional (int -> real -> complex), so
        /// promoted_data_type(kReal) *rejects* a kComplex input.  This helper
        /// additionally performs the narrowing complex -> real (taking the
        /// real part), which is not a promotion at all.  It also absorbs the
        /// unit multiplier, which promoted_data_type() deliberately does not.
        ///
        /// - kReal / kInteger: value * unit().multiplier()
        /// - kComplex:         real part * unit().multiplier()
        /// - kString / kBoolean: throws std::invalid_argument
        /// - vector / matrix:    throws std::logic_error
        ///
        /// This is the unit-safe way to compare a Measurement against a
        /// canonicalised DataSeries axis.  A bare as_scalar<double>() silently
        /// ignores the unit and mis-compares (e.g. 500 mV reads as 500, not
        /// 0.5, and never matches a 0.5 V axis).
        double promoted_double() const;

    private:
        void infer_metadata();


        // ---- transform helpers ------------------------------------------
        //
        //  Each dispatch switches on data_type_ and calls the typed impl.
        //  All switch alternatives are instantiated at compile time;
        //  the int/long overload trick picks the int overload when Func(T)
        //  is valid (SFINAE), otherwise the long overload silently returns
        //  *this.  At runtime only the matching data_type_ branch executes.

        // Scalar dispatch
        template <typename Func>
        Measurement transform_scalar_dispatch(Func&& func) const {
            switch (data_type_) {
                case DataType::kReal:
                    return transform_scalar_impl<double>(std::forward<Func>(func), 0);
                case DataType::kInteger:
                    return transform_scalar_impl<int>(std::forward<Func>(func), 0);
                case DataType::kComplex:
                    return transform_scalar_impl<std::complex<double>>(std::forward<Func>(func), 0);
                case DataType::kString:
                    return transform_scalar_impl<std::string>(std::forward<Func>(func), 0);
                default:
                    return *this;
            }
        }

        // Vector dispatch
        template <typename Func>
        Measurement transform_vector_dispatch(Func&& func) const {
            switch (data_type_) {
                case DataType::kReal:
                    return transform_vector_impl<double>(std::forward<Func>(func), 0);
                case DataType::kInteger:
                    return transform_vector_impl<int>(std::forward<Func>(func), 0);
                case DataType::kComplex:
                    return transform_vector_impl<std::complex<double>>(std::forward<Func>(func), 0);
                default:
                    return *this;
            }
        }

        // Matrix dispatch
        template <typename Func>
        Measurement transform_matrix_dispatch(Func&& func) const {
            switch (data_type_) {
                case DataType::kReal:
                    return transform_matrix_impl<double>(std::forward<Func>(func), 0);
                case DataType::kInteger:
                    return transform_matrix_impl<int>(std::forward<Func>(func), 0);
                case DataType::kComplex:
                    return transform_matrix_impl<std::complex<double>>(std::forward<Func>(func), 0);
                default:
                    return *this;
            }
        }

        // -- SFINAE scalar (int priority) ---------------------------------
        //  Uses expression SFINAE on a non-type template parameter so that
        //  Measurement does not appear inside a decltype within the class
        //  body (avoids IntelliSense "incomplete type" false positives).

        template <typename T, typename Func,
            decltype(std::declval<Func>()(std::declval<const T&>()), 0) = 0>
        Measurement transform_scalar_impl(Func&& func, int) const
        {
            return Scalar(func(boost::get<T>(storage_)), unit_);
        }

        template <typename T, typename Func>
        Measurement transform_scalar_impl(Func&& func, long) const {
            (void)func; return *this;
        }

        // -- SFINAE vector (int priority) ---------------------------------

        template <typename T, typename Func,
            decltype(std::declval<Func>()(std::declval<const T&>()), 0) = 0>
        Measurement transform_vector_impl(Func&& func, int) const
        {
            typedef decltype(func(std::declval<const T&>())) Out;
            Index w = shape_[0];
            Eigen::Matrix<Out, 1, Eigen::Dynamic> v(w);
            const auto& src = boost::get<Eigen::Matrix<T, 1, Eigen::Dynamic>>(storage_);
            for (Index i = 0; i < w; ++i) v(i) = func(src(i));
            return Measurement::Vector(v, unit_);
        }

        template <typename T, typename Func>
        Measurement transform_vector_impl(Func&& func, long) const {
            (void)func; return *this;
        }

        // -- SFINAE matrix (int priority) ---------------------------------

        template <typename T, typename Func,
            decltype(std::declval<Func>()(std::declval<const T&>()), 0) = 0>
        Measurement transform_matrix_impl(Func&& func, int) const
        {
            typedef decltype(func(std::declval<const T&>())) Out;
            Index r = shape_[0], c = shape_[1];
            Eigen::Matrix<Out, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(r, c);
            const auto& src = boost::get<
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(storage_);
            for (Index i = 0; i < r; ++i)
                for (Index j = 0; j < c; ++j)
                    m(i, j) = func(src(i, j));
            return Measurement::Matrix(m, unit_);
        }

        template <typename T, typename Func>
        Measurement transform_matrix_impl(Func&& func, long) const {
            (void)func; return *this;
        }


        DataType             data_type_;
        DataShape            shape_;
        Storage              storage_;
        Unit                 unit_;
    };

    // =========================================================================
    // Template implementation
    // =========================================================================

    // -- as_scalar<T> ------------------------------------------------------------

    template <typename T>
    T Measurement::as_scalar() const
    {
        if (shape_.kind() != DataKind::kScalar)
            throw std::logic_error("as_scalar: Measurement is not a scalar (kind=" +
                std::to_string(static_cast<int>(shape_.kind())) + ")");
        return boost::get<T>(storage_);
    }

    // -- as_vector<T> (numeric) --------------------------------------------------

    template <typename T>
    typename std::enable_if<
        !std::is_same<T, std::string>::value,
        VecConstMap<T>>::type
    Measurement::as_vector() const
    {
        if (shape_.kind() != DataKind::kVector)
            throw std::logic_error("as_vector: Measurement is not a vector (kind=" +
                std::to_string(static_cast<int>(shape_.kind())) + ")");
        typedef Vec<T> VecType;
        const VecType& vec = boost::get<VecType>(storage_);
        return VecConstMap<T>(vec.data(), 1, vec.size());
    }

    // -- as_vector<T> (string) ---------------------------------------------------

    template <typename T>
    typename std::enable_if<
        std::is_same<T, std::string>::value,
        const VecXs&>::type
    Measurement::as_vector() const
    {
        if (shape_.kind() != DataKind::kVector)
            throw std::logic_error("as_vector: Measurement is not a vector (kind=" +
                std::to_string(static_cast<int>(shape_.kind())) + ")");
        return boost::get<VecXs>(storage_);
    }

    // -- as_matrix<T> (numeric) --------------------------------------------------

    template <typename T>
    typename std::enable_if<
        !std::is_same<T, std::string>::value,
        MatConstMap<T>>::type
    Measurement::as_matrix() const
    {
        if (shape_.kind() != DataKind::kMatrix)
            throw std::logic_error("as_matrix: Measurement is not a matrix (kind=" +
                std::to_string(static_cast<int>(shape_.kind())) + ")");
        typedef Mat<T> MatType;
        const MatType& mat = boost::get<MatType>(storage_);
        return MatConstMap<T>(mat.data(), mat.rows(), mat.cols());
    }

    // -- as_matrix<T> (string) ---------------------------------------------------

    template <typename T>
    typename std::enable_if<
        std::is_same<T, std::string>::value,
        const MatXs&>::type
    Measurement::as_matrix() const
    {
        if (shape_.kind() != DataKind::kMatrix)
            throw std::logic_error("as_matrix: Measurement is not a matrix (kind=" +
                std::to_string(static_cast<int>(shape_.kind())) + ")");
        return boost::get<MatXs>(storage_);
    }

    // =========================================================================
    //  Measurement::Scalar -- scalar factory dispatch by element type
    // =========================================================================

    template <>
    inline Measurement Measurement::Scalar<double>(const double& v,
                                                   const Unit& u)
    {
        return Measurement::Real(v, u);
    }

    template <>
    inline Measurement Measurement::Scalar<int>(const int& v,
                                                const Unit& u)
    {
        return Measurement::Integer(v, u);
    }

    template <>
    inline Measurement Measurement::Scalar<std::complex<double> >(
        const std::complex<double>& v, const Unit& u)
    {
        return Measurement::Complex(v, u);
    }

    template <>
    inline Measurement Measurement::Scalar<std::string>(
        const std::string& v, const Unit&)
    {
        return Measurement::String(v);   // strings carry no unit
    }

} // namespace xdataset

#endif // MEASUREMENT_H
