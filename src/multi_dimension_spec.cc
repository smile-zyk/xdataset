#include "multi_dimension_spec.h"
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace xdataset
{
    MultiDimensionSpec::MultiDimensionSpec()
        : dims_()
    {
    }

    MultiDimensionSpec::MultiDimensionSpec(const std::vector<DimensionSpec>& dims)
        : dims_(dims)
    {
    }

    MultiDimensionSpec& MultiDimensionSpec::add_uniform(std::size_t size)
    {
        dims_.push_back(DimensionSpec::Uniform(size));
        return *this;
    }

    MultiDimensionSpec& MultiDimensionSpec::add_jagged(const std::vector<std::size_t>& sizes)
    {
        std::size_t expected_count = compute_cell_count();
        if (sizes.size() != expected_count)
        {
            throw std::invalid_argument("jagged sizes count must match the total cell count of previous dimensions");
        }
        dims_.push_back(DimensionSpec::Jagged(sizes));
        return *this;
    }

    MultiDimensionSpec& MultiDimensionSpec::add_dimension(const DimensionSpec& dim)
    {
        dims_.push_back(dim);
        return *this;
    }

    std::size_t MultiDimensionSpec::rank() const
    {
        return dims_.size();
    }

    std::size_t MultiDimensionSpec::ndim() const
    {
        return rank();
    }

    bool MultiDimensionSpec::empty() const
    {
        return dims_.empty();
    }

    const std::vector<DimensionSpec>& MultiDimensionSpec::dims() const
    {
        return dims_;
    }

    const DimensionSpec& MultiDimensionSpec::dim(std::size_t index) const
    {
        if (index >= dims_.size())
        {
            throw std::out_of_range("dimension index out of range");
        }
        return dims_[index];
    }

    void MultiDimensionSpec::set_dimensions(const std::vector<DimensionSpec>& dims)
    {
        dims_ = dims;
    }

    std::size_t MultiDimensionSpec::flat_index(const std::vector<std::size_t>& multi_index) const
    {
        if (multi_index.size() != dims_.size())
        {
            throw std::invalid_argument("multi_index size must match rank");
        }
        
        if (dims_.empty())
        {
            return 0;
        }

        std::size_t flat = 0;
        for (std::size_t d = 0; d < dims_.size(); ++d)
        {
            if (dims_[d].is_uniform())
            {
                // For uniform dimension: flat = flat * size_d + index_d (row-major order)
                flat = flat * static_cast<std::size_t>(dims_[d].as_uniform()->size) + multi_index[d];
            }
            else
            {
                // For jagged dimension: flat is the parent row index, child is multi_index[d]
                const std::vector<std::size_t>& prefix = dims_[d].prefix_sum();
                flat = prefix[flat] + multi_index[d];
            }
        }
        return flat;
    }

    std::vector<std::size_t> MultiDimensionSpec::multi_index(std::size_t flat) const
    {
        std::vector<std::size_t> result(dims_.size(), 0);
        
        if (dims_.empty())
        {
            if (flat != 0)
            {
                throw std::out_of_range("flat index out of bounds");
            }
            return result;
        }

        std::size_t residual = flat;
        for (std::size_t d = dims_.size(); d-- > 0;)
        {
            if (dims_[d].is_uniform())
            {
                const std::size_t size = static_cast<std::size_t>(dims_[d].as_uniform()->size);
                result[d] = residual % size;
                residual = residual / size;
            }
            else
            {
                // For jagged: binary search in prefix sum to find the parent row,
                // then compute the child index within that parent.
                const std::vector<std::size_t>& prefix = dims_[d].prefix_sum();
                auto it = std::upper_bound(prefix.begin(), prefix.end(), residual);
                std::size_t parent_row = static_cast<std::size_t>(it - prefix.begin()) - 1;
                result[d] = residual - prefix[parent_row];
                residual = parent_row;
            }
        }
        return result;
    }

    std::size_t MultiDimensionSpec::compute_cell_count() const
    {
        std::size_t count = 1;
        for (std::size_t i = 0; i < dims_.size(); ++i)
        {
            if (dims_[i].is_uniform())
            {
                count *= static_cast<std::size_t>(dims_[i].as_uniform()->size);
            }
            else
            {
                // For jagged, directly set to total (no multiply, jagged defines the structure)
                count = dims_[i].prefix_sum().back();
            }
        }
        return count;
    }

    void MultiDimensionSpec::for_each_leaf_row(const LeafRowVisitor& visitor) const
    {
        for_each_leaf_row(visitor, 0, compute_cell_count());
    }

    void MultiDimensionSpec::for_each_leaf_row(const LeafRowVisitor& visitor, std::size_t start_flat_row, std::size_t end_flat_row) const
    {
        if (!visitor)
        {
            return;
        }

        const std::size_t total = compute_cell_count();
        if (start_flat_row >= end_flat_row || start_flat_row >= total)
        {
            return;
        }
        if (end_flat_row > total)
        {
            end_flat_row = total;
        }

        if (dims_.empty())
        {
            LeafRow leaf_row;
            visitor(leaf_row);
            return;
        }

        const std::size_t rank = dims_.size();
        LeafRow leaf_row;
        leaf_row.multi_index.resize(rank);
        leaf_row.dimension_row_indices.resize(rank, 0);

        for (std::size_t flat = start_flat_row; flat < end_flat_row; ++flat)
        {
            leaf_row.multi_index = multi_index(flat);
            leaf_row.row_flat = flat;

            // Compute dimension_row_indices in a single O(rank) pass.
            std::size_t parent_flat = 0;
            for (std::size_t d = 0; d < rank; ++d)
            {
                if (dims_[d].is_uniform())
                {
                    leaf_row.dimension_row_indices[d] = leaf_row.multi_index[d];
                    parent_flat = parent_flat * static_cast<std::size_t>(dims_[d].uniform_size())
                        + leaf_row.multi_index[d];
                }
                else
                {
                    std::size_t start = 0;
                    std::size_t end = 0;
                    dims_[d].child_range(parent_flat, start, end);
                    (void)end;
                    leaf_row.dimension_row_indices[d] = start + leaf_row.multi_index[d];
                    parent_flat = start + leaf_row.multi_index[d];
                }
            }

            visitor(leaf_row);
        }
    }

} // namespace xdataset
