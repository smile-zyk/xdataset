#ifndef XDATASET_PREDEFINE_H
#define XDATASET_PREDEFINE_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <complex>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
//  Windows DLL export / import
// ---------------------------------------------------------------------------
#ifdef _WIN32
  #ifdef XDATASET_BUILD_DLL
    #define XDATASET_API __declspec(dllexport)
  #else
    #define XDATASET_API __declspec(dllimport)
  #endif
#else
  #define XDATASET_API
#endif

namespace xdataset
{
    using Index = Eigen::Index;

    // =========================================================================
    //  Convenient Eigen type aliases (all RowMajor for cache-friendly storage)
    // =========================================================================

    // --- Template aliases ---
    template <typename T>
    using Vec = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

    template <typename T>
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    template <typename T>
    using VecMap = Eigen::Map<Vec<T>>;
    template <typename T>
    using VecConstMap = Eigen::Map<const Vec<T>>;
    template <typename T>
    using MatMap = Eigen::Map<Mat<T>>;
    template <typename T>
    using MatConstMap = Eigen::Map<const Mat<T>>;

    // --- Concrete numeric row-vector types ---
    using VecXd  = Vec<double>;
    using VecXi  = Vec<int>;
    using VecXcd = Vec<std::complex<double>>;

    // --- Concrete numeric matrix types ---
    using MatXd  = Mat<double>;
    using MatXi  = Mat<int>;
    using MatXcd = Mat<std::complex<double>>;

    // --- String tensor types ---
    using VecXs = Eigen::Tensor<std::string, 1>;
    using MatXs = Eigen::Tensor<std::string, 2>;

    enum class DataKind
    {
        kScalar,
        kVector,
        kMatrix
    };

    enum class DataType
    {
        kReal,
        kInteger,
        kComplex,
        kString,
        kBoolean
    };

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
    struct DataTypeOf;

    template <>
    struct DataTypeOf<double> {
        static const DataType tag = DataType::kReal;
    };

    template <>
    struct DataTypeOf<int> {
        static const DataType tag = DataType::kInteger;
    };

    template <>
    struct DataTypeOf<std::complex<double> > {
        static const DataType tag = DataType::kComplex;
    };

    template <>
    struct DataTypeOf<std::string> {
        static const DataType tag = DataType::kString;
    };

    /// Render a DataType as "Double" / "Integer" / "Complex" / "String" / "Boolean".
    inline const char* DataTypeToString(DataType type)
    {
        switch (type)
        {
            case DataType::kInteger: return "Integer";
            case DataType::kReal:    return "Real";
            case DataType::kComplex: return "Complex";
            case DataType::kString:  return "String";
            case DataType::kBoolean: return "Boolean";
        }
        return "Unknown";
    }

    /// True when `name` is a valid REL identifier: [A-Za-z_][A-Za-z0-9_]*.
    /// Dataset names and Block path segments must all satisfy this so that
    /// REL references (dataset.block.array) and the global source_path
    /// ("<dataset>/<block path>") remain unambiguous.
    inline bool IsValidIdentifier(const std::string& name)
    {
        if (name.empty()) return false;
        const char c0 = name[0];
        const bool start_ok =
            (c0 >= 'a' && c0 <= 'z') || (c0 >= 'A' && c0 <= 'Z') || c0 == '_';
        if (!start_ok) return false;
        for (std::size_t i = 1; i < name.size(); ++i)
        {
            const char c = name[i];
            const bool ok =
                (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_';
            if (!ok) return false;
        }
        return true;
    }

    /// Block paths use '.' as the separator between segments, and between the
    /// Dataset name and a Block path in the source_path (e.g.
    /// "noise.simulation.SP1.SP"), so they are REL-compatible.
    ///
    /// On-disk formats are NOT affected by this choice:
    ///   - HDF5 stores the hierarchy as nested groups (HDF5 uses its own '/'
    ///     separator at the file level); both the writer (SplitPath) and the
    ///     reader (path join) derive the in-memory path from a single
    ///     consistent separator, so round-trips are preserved.
    ///   - Touchstone uses a single-segment block ("SP"), so it never splits
    ///     a multi-segment path.
}

#endif // XDATASET_PREDEFINE_H
