#ifndef XDATASET_VARIABLE_DATA_H_
#define XDATASET_VARIABLE_DATA_H_

#include "cell_series.h"

#include <numeric>
#include <stdexcept>
#include <vector>

namespace xdataset {

class VariableData {
public:
    // Create empty VariableData
    VariableData() = default;

    // Create VariableData with shape and CellSeries
    // The product of shape dimensions must equal series size
    VariableData(const std::vector<Index>& shape, const CellSeries& series)
        : shape_(shape), series_(series) {
        validate_shape();
    }

    // Create VariableData with shape, dtype, and fill value
    template <typename T>
    static VariableData Create(const std::vector<Index>& shape, const T& fill_val = T()) {
        std::size_t size = compute_size(shape);
        CellSeries s = CellSeries::Scalars<T>(size, fill_val);
        return VariableData(shape, s);
    }

    // Get the shape
    const std::vector<Index>& shape() const { return shape_; }

    // Get the underlying series
    const CellSeries& series() const { return series_; }
    CellSeries& series() { return series_; }

    // Get total number of elements
    std::size_t size() const { return series_.size(); }

    // Get number of dimensions
    std::size_t rank() const { return shape_.size(); }

    // Convert multi-dimensional indices to linear index (row-major order)
    // For example, with shape [3, 5], index (1, 2) maps to 1*5 + 2 = 7
    std::size_t flat_index(const std::vector<Index>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("number of indices must match rank");
        }

        std::size_t idx = 0;
        std::size_t stride = 1;
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            if (indices[d] < 0 || indices[d] >= shape_[d]) {
                throw std::out_of_range("index out of bounds");
            }
            idx += indices[d] * stride;
            stride *= shape_[d];
        }
        return idx;
    }

    // Access scalar at multi-dimensional index
    template <typename T>
    T& at(const std::vector<Index>& indices) {
        return series_.scalar_at<T>(flat_index(indices));
    }

    template <typename T>
    const T& at(const std::vector<Index>& indices) const {
        return series_.scalar_at<T>(flat_index(indices));
    }

    // Convenience: access 1D index
    template <typename T>
    T& at_1d(Index i) {
        return series_.scalar_at<T>(i);
    }

    template <typename T>
    const T& at_1d(Index i) const {
        return series_.scalar_at<T>(i);
    }

    // Convenience: access 2D index (i, j)
    template <typename T>
    T& at_2d(Index i, Index j) {
        std::vector<Index> indices = {i, j};
        return at<T>(indices);
    }

    template <typename T>
    const T& at_2d(Index i, Index j) const {
        std::vector<Index> indices = {i, j};
        return at<T>(indices);
    }

    // Convenience: access 3D index (i, j, k)
    template <typename T>
    T& at_3d(Index i, Index j, Index k) {
        std::vector<Index> indices = {i, j, k};
        return at<T>(indices);
    }

    template <typename T>
    const T& at_3d(Index i, Index j, Index k) const {
        std::vector<Index> indices = {i, j, k};
        return at<T>(indices);
    }

private:
    static std::size_t compute_size(const std::vector<Index>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                               [](std::size_t a, Index b) { return a * b; });
    }

    void validate_shape() const {
        if (shape_.empty()) return;
        std::size_t expected_size = compute_size(shape_);
        if (expected_size != series_.size()) {
            throw std::invalid_argument(
                "shape dimensions product must equal series size");
        }
    }

    std::vector<Index> shape_;
    CellSeries series_;
};

}  // namespace xdataset

#endif  // XDATASET_VARIABLE_DATA_H_
