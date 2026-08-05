#ifndef XDATASET_PREDEFINE_H
#define XDATASET_PREDEFINE_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <complex>
#include <string>
#include <vector>

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

    /// Render a DataType as "Double" / "Integer" / "Complex" / "String" / "Boolean".
    inline const char* DataTypeToString(DataType type)
    {
        switch (type)
        {
            case DataType::kInteger: return "Integer";
            case DataType::kReal:    return "Double";
            case DataType::kComplex: return "Complex";
            case DataType::kString:  return "String";
            case DataType::kBoolean: return "Boolean";
        }
        return "Unknown";
    }
}

#endif // XDATASET_PREDEFINE_H