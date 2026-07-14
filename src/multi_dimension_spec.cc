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
                // For jagged dimension: use prefix sum to index into next level
                const std::vector<std::size_t>& prefix = dims_[d].prefix_sum();
                flat = prefix[multi_index[d]];
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
                const std::size_t stride = compute_cell_count_from(d + 1);
                result[d] = residual / stride;
                residual = residual % stride;
            }
            else
            {
                // For jagged: binary search in prefix sum
                const std::vector<std::size_t>& prefix = dims_[d].prefix_sum();
                auto it = std::upper_bound(prefix.begin(), prefix.end(), residual);
                result[d] = static_cast<std::size_t>((it - prefix.begin()) - 1);
                residual = residual - prefix[result[d]];
            }
        }
        return result;
    }

    std::size_t MultiDimensionSpec::compute_cell_count_from(std::size_t start_dim) const
    {
        if (start_dim >= dims_.size())
        {
            return 1;
        }

        std::size_t count = 1;
        for (std::size_t i = start_dim; i < dims_.size(); ++i)
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

    std::size_t MultiDimensionSpec::compute_cell_count() const
    {
        return compute_cell_count_from(0);
    }

    void MultiDimensionSpec::for_each_leaf_row(const LeafRowVisitor& visitor) const
    {
        if (!visitor)
        {
            return;
        }

        if (dims_.empty())
        {
            LeafRow leaf_row;
            visitor(leaf_row);
            return;
        }

        std::vector<std::size_t> multi_index(dims_.size(), 0);
        std::vector<std::size_t> source_rows(dims_.size(), 0);

        std::function<void(std::size_t, std::size_t)> walk =
            [&](std::size_t dim_idx, std::size_t parent_flat)
        {
            std::size_t child_count = 0;
            if (dims_[dim_idx].is_uniform())
            {
                child_count = static_cast<std::size_t>(dims_[dim_idx].uniform_size());
            }
            else
            {
                child_count = dims_[dim_idx].child_width(parent_flat);
            }

            for (std::size_t idx = 0; idx < child_count; ++idx)
            {
                multi_index[dim_idx] = idx;

                std::size_t current_flat = 0;
                if (dims_[dim_idx].is_uniform())
                {
                    const std::size_t size = static_cast<std::size_t>(dims_[dim_idx].uniform_size());
                    current_flat = parent_flat * size + idx;
                    source_rows[dim_idx] = idx;
                }
                else
                {
                    std::size_t start = 0;
                    std::size_t end = 0;
                    dims_[dim_idx].child_range(parent_flat, start, end);
                    (void)end;
                    current_flat = start + idx;
                    source_rows[dim_idx] = current_flat;
                }

                if (dim_idx + 1 < dims_.size())
                {
                    walk(dim_idx + 1, current_flat);
                    continue;
                }

                LeafRow leaf_row;
                leaf_row.multi_index = multi_index;
                leaf_row.dimension_row_indices = source_rows;
                leaf_row.row_flat = current_flat;
                visitor(leaf_row);
            }
        };

        walk(0, 0);
    }

} // namespace xdataset
