#ifndef XDATASET_PREDEFINE_H
#define XDATASET_PREDEFINE_H

#include <Eigen/Dense>
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
        kBoolean,
        kNull
    };

    struct DataShape {
        std::vector<Index> dims;

        DataShape() = default;
        DataShape(std::initializer_list<Index> il) : dims(il) {}
        explicit DataShape(const std::vector<Index>& v) : dims(v) {}

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