#include "multi_dimension_spec.h"
#include <stdexcept>
#include <algorithm>

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

    MultiDimensionSpec& MultiDimensionSpec::add_uniform(Index size)
    {
        dims_.push_back(DimensionSpec::Uniform(size));
        return *this;
    }

    MultiDimensionSpec& MultiDimensionSpec::add_jagged(const std::vector<Index>& sizes)
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

    MultiDimensionSpec MultiDimensionSpec::select(const std::vector<MultiIndexSelector>& selectors) const
    {
        if (selectors.size() != dims_.size())
        {
            throw std::invalid_argument("number of selectors must match rank");
        }

        MultiDimensionSpec result;
        for (std::size_t d = 0; d < selectors.size(); ++d)
        {
            const MultiIndexSelector& selector = selectors[d];
            
            if (selector.is_any())
            {
                // Retain the original dimension
                result.add_dimension(dims_[d]);
            }
            else if (selector.is_equal())
            {
                // Drop this dimension (skip it)
                // No action needed, just continue to next dimension
            }
            else if (selector.is_in())
            {
                // Convert to jagged dimension with sizes equal to number of selected indices
                const std::vector<Index>& indices = selector.in_values();
                std::vector<Index> sizes(indices.size(), 1);
                result.add_jagged(sizes);
            }
        }

        return result;
    }

} // namespace xdataset
