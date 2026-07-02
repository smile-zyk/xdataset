#ifndef XDATASET_VARIABLE_DATA_H_
#define XDATASET_VARIABLE_DATA_H_

#include "cell_series.h"

#include <numeric>
#include <stdexcept>
#include <variant>
#include <vector>

namespace xdataset {

// Represents a shape that may contain both regular (fixed-size) and jagged (variable-size) dimensions
// Example: [3, 2, [2,3,6,8,1,2], 3]
//   - Dimension 0: regular (3)
//   - Dimension 1: regular (2)
//   - Dimension 2: jagged with 6 elements of sizes 2,3,6,8,1,2
//   - Dimension 3: regular (3)
//   - Total elements: 3 * 2 * 6 * 3 = 108 entries, but each entry in dim 2 has variable size in dim 3
class ShapeSpec {
public:
    // Each dimension can be regular (Index) or jagged (vector of sizes)
    using DimSpec = std::variant<Index, std::vector<Index>>;

    ShapeSpec() = default;

    // Construct from vector of dimension specifications
    explicit ShapeSpec(const std::vector<DimSpec>& dims) : dims_(dims) {}

    // Builder: add regular dimension
    ShapeSpec& add_regular(Index size) {
        dims_.emplace_back(size);
        return *this;
    }

    // Builder: add jagged dimension
    ShapeSpec& add_jagged(const std::vector<Index>& sizes) {
        dims_.emplace_back(sizes);
        return *this;
    }

    // Get number of dimensions
    std::size_t rank() const { return dims_.size(); }

    // Get the dimension specification
    const std::vector<DimSpec>& dims() const { return dims_; }

    // Check if dimension is jagged
    bool is_jagged(std::size_t dim) const {
        return std::holds_alternative<std::vector<Index>>(dims_[dim]);
    }

    // Get size of dimension (for regular: fixed size; for jagged: number of elements)
    Index dim_size(std::size_t dim) const {
        if (std::holds_alternative<Index>(dims_[dim])) {
            return std::get<Index>(dims_[dim]);
        } else {
            return static_cast<Index>(std::get<std::vector<Index>>(dims_[dim]).size());
        }
    }

    // Compute total number of elements
    // For jagged dims: multiply by element count, then account for variable inner sizes
    std::size_t total_size() const {
        if (dims_.empty()) return 0;
        std::size_t total = 1;
        for (const auto& dim : dims_) {
            if (std::holds_alternative<Index>(dim)) {
                total *= std::get<Index>(dim);
            } else {
                // For jagged dimension: product of all variable sizes
                const auto& sizes = std::get<std::vector<Index>>(dim);
                std::size_t product = std::accumulate(
                    sizes.begin(), sizes.end(), std::size_t(1),
                    [](std::size_t a, Index b) { return a * b; });
                total *= product;
            }
        }
        return total;
    }

    // Validate multi-dimensional indices against shape
    void validate_indices(const std::vector<Index>& indices) const {
        if (indices.size() != dims_.size()) {
            throw std::invalid_argument("number of indices must match rank");
        }
        for (std::size_t d = 0; d < dims_.size(); ++d) {
            Index idx = indices[d];
            if (idx < 0) {
                throw std::out_of_range("index must be non-negative");
            }
            if (std::holds_alternative<Index>(dims_[d])) {
                Index size = std::get<Index>(dims_[d]);
                if (idx >= size) {
                    throw std::out_of_range("index out of bounds in regular dimension");
                }
            } else {
                const auto& sizes = std::get<std::vector<Index>>(dims_[d]);
                if (idx >= static_cast<Index>(sizes.size())) {
                    throw std::out_of_range("index out of bounds in jagged dimension");
                }
            }
        }
    }

    // Compute flat index from multi-dimensional indices (row-major order)
    // For jagged dimensions: account for variable sizes
    std::size_t flat_index(const std::vector<Index>& indices) const {
        validate_indices(indices);

        std::size_t idx = 0;
        std::size_t stride = 1;

        // Process dimensions from last to first (row-major)
        for (int d = static_cast<int>(dims_.size()) - 1; d >= 0; --d) {
            Index idx_d = indices[d];

            if (std::holds_alternative<Index>(dims_[d])) {
                // Regular dimension
                Index size = std::get<Index>(dims_[d]);
                idx += idx_d * stride;
                stride *= size;
            } else {
                // Jagged dimension: accumulate sizes of previous elements
                const auto& sizes = std::get<std::vector<Index>>(dims_[d]);
                std::size_t offset = 0;
                for (int j = 0; j < idx_d; ++j) {
                    offset += sizes[j];
                }
                idx += offset * stride;
                stride *= sizes[idx_d];
            }
        }
        return idx;
    }

    std::vector<DimSpec> dims_;
};

class VariableData {
public:
    // Create empty VariableData
    VariableData() = default;

    // Create VariableData with regular shape and CellSeries
    // For backward compatibility: convert std::vector<Index> to ShapeSpec with all regular dims
    VariableData(const std::vector<Index>& shape, const CellSeries& series)
        : shape_spec_(), series_(series) {
        for (Index size : shape) {
            shape_spec_.add_regular(size);
        }
        validate_shape();
    }

    // Create VariableData with ShapeSpec and CellSeries
    VariableData(const ShapeSpec& shape, const CellSeries& series)
        : shape_spec_(shape), series_(series) {
        validate_shape();
    }

    // Create VariableData with regular shape, dtype, and fill value
    template <typename T>
    static VariableData Create(const std::vector<Index>& shape, const T& fill_val = T()) {
        ShapeSpec spec;
        for (Index size : shape) {
            spec.add_regular(size);
        }
        std::size_t size = spec.total_size();
        CellSeries s = CellSeries::Scalars<T>(size, fill_val);
        return VariableData(spec, s);
    }

    // Create VariableData with ShapeSpec, dtype, and fill value
    template <typename T>
    static VariableData Create(const ShapeSpec& shape, const T& fill_val = T()) {
        std::size_t size = shape.total_size();
        CellSeries s = CellSeries::Scalars<T>(size, fill_val);
        return VariableData(shape, s);
    }

    // Get the shape specification
    const ShapeSpec& shape_spec() const { return shape_spec_; }

    // Get the underlying series
    const CellSeries& series() const { return series_; }
    CellSeries& series() { return series_; }

    // Get total number of elements
    std::size_t size() const { return series_.size(); }

    // Get number of dimensions
    std::size_t rank() const { return shape_spec_.rank(); }

    // Access scalar at multi-dimensional index
    template <typename T>
    T& at(const std::vector<Index>& indices) {
        return series_.scalar_at<T>(shape_spec_.flat_index(indices));
    }

    template <typename T>
    const T& at(const std::vector<Index>& indices) const {
        return series_.scalar_at<T>(shape_spec_.flat_index(indices));
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

    // Convenience: access 4D index (i, j, k, l)
    template <typename T>
    T& at_4d(Index i, Index j, Index k, Index l) {
        std::vector<Index> indices = {i, j, k, l};
        return at<T>(indices);
    }

    template <typename T>
    const T& at_4d(Index i, Index j, Index k, Index l) const {
        std::vector<Index> indices = {i, j, k, l};
        return at<T>(indices);
    }

private:
    void validate_shape() const {
        if (shape_spec_.rank() == 0) return;
        std::size_t expected_size = shape_spec_.total_size();
        if (expected_size != series_.size()) {
            throw std::invalid_argument(
                "shape dimensions product must equal series size");
        }
    }

    ShapeSpec shape_spec_;
    CellSeries series_;
};

}  // namespace xdataset

#endif  // XDATASET_VARIABLE_DATA_H_
