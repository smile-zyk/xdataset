#ifndef XDATASET_PREDEFINE_H
#define XDATASET_PREDEFINE_H

#include <Eigen/Dense>

namespace xdataset
{
    using Index = Eigen::Index;

    enum class DataKind
    {
        kScalar,
        kVector,
        kMatrix
    };

    enum class DTypeTag
    {
        kReal,
        kInteger,
        kComplex,
        kString
    };
}

#endif // XDATASET_PREDEFINE_H