#include "data_shape.h"

#include <stdexcept>

namespace xdataset {

DataKind DataShape::kind() const
{
    if (dims.empty())    return DataKind::kScalar;
    if (dims.size() == 1) return DataKind::kVector;
    return DataKind::kMatrix;
}

Index DataShape::element_count() const
{
    if (dims.empty())    return 1;
    if (dims.size() == 1) return dims[0];
    return dims[0] * dims[1];
}

std::vector<Index> DataShape::element_position(Index e) const
{
    std::vector<Index> out;
    switch (kind()) {
        case DataKind::kScalar:
            break;
        case DataKind::kVector:
            out.push_back(e);
            break;
        case DataKind::kMatrix: {
            const Index cols = dims[1];
            out.push_back(e / cols);
            out.push_back(e % cols);
            break;
        }
    }
    return out;
}

Index DataShape::element_index(const std::vector<Index>& pos) const
{
    switch (kind()) {
        case DataKind::kScalar:
            if (!pos.empty())
                throw std::invalid_argument(
                    "DataShape::element_index: scalar expects an empty position");
            return 0;
        case DataKind::kVector:
            if (pos.size() != 1)
                throw std::invalid_argument(
                    "DataShape::element_index: vector expects 1 index, got " +
                    std::to_string(pos.size()));
            return pos[0];
        case DataKind::kMatrix:
            if (pos.size() != 2)
                throw std::invalid_argument(
                    "DataShape::element_index: matrix expects 2 indices, got " +
                    std::to_string(pos.size()));
            return pos[0] * dims[1] + pos[1];
    }
    return 0;
}

std::vector<Index> DataShape::copy() const
{
    return dims;
}

std::string DataShape::to_string() const
{
    switch (kind()) {
        case DataKind::kScalar: return "Scalar";
        case DataKind::kVector: return "Vector(" + std::to_string(dims[0]) + ")";
        case DataKind::kMatrix: return "Matrix(" + std::to_string(dims[0]) + ", " + std::to_string(dims[1]) + ")";
    }
    return "Unknown";
}

} // namespace xdataset
