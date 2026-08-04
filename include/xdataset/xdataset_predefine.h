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

    struct DataShape {
        std::vector<Index> dims;

        DataShape() = default;
        DataShape(std::initializer_list<Index> il) : dims(il) {}
        explicit DataShape(const std::vector<Index>& v) : dims(v) {}

        // ------------------------------------------------------------------
        //  Named static constructors
        // ------------------------------------------------------------------

        static DataShape Scalar()            { return DataShape{}; }
        static DataShape Vector(Index width) { return DataShape{width}; }
        static DataShape Matrix(Index rows, Index cols) { return DataShape{rows, cols}; }

        // ------------------------------------------------------------------

        Index& operator[](size_t i)             { return dims[i]; }
        Index  operator[](size_t i) const       { return dims[i]; }
        size_t size()                     const { return dims.size(); }
        bool   empty()                    const { return dims.empty(); }
        void   clear()                          { dims.clear(); }
        void   push_back(Index v)               { dims.push_back(v); }

        bool operator==(const DataShape& o) const { return dims == o.dims; }
        bool operator!=(const DataShape& o) const { return dims != o.dims; }

        DataKind kind() const {
            if (dims.empty())    return DataKind::kScalar;
            if (dims.size() == 1) return DataKind::kVector;
            return DataKind::kMatrix;
        }

        Index element_count() const {
            if (dims.empty())    return 1;
            if (dims.size() == 1) return dims[0];
            return dims[0] * dims[1];
        }

        std::vector<Index> copy() const { return dims; }
    };
}

#endif // XDATASET_PREDEFINE_H