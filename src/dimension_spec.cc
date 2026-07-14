#include "dimension_spec.h"
#include <cstddef>
#include <stdexcept>

namespace xdataset
{
    // Factory methods
    DimensionSpec DimensionSpec::Uniform(Index size)
    {
        if (size < 0)
        {
            throw std::invalid_argument("uniform dimension size must be non-negative");
        }
        return DimensionSpec(UniformDim(size));
    }

    DimensionSpec DimensionSpec::Jagged(const std::vector<std::size_t>& sizes)
    {
        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            if (sizes[i] < 0)
            {
                throw std::invalid_argument("jagged dimension size must be non-negative");
            }
        }
        return DimensionSpec(JaggedDim(sizes));
    }

    // Constructors
    DimensionSpec::DimensionSpec(const UniformDim& uniform)
        : spec_(uniform)
    {
    }

    DimensionSpec::DimensionSpec(const JaggedDim& jagged)
        : spec_(jagged)
    {
    }

    // Type checking
    bool DimensionSpec::is_uniform() const
    {
        return spec_.which() == 0;  // 0 = UniformDim, 1 = JaggedDim
    }

    bool DimensionSpec::is_jagged() const
    {
        return spec_.which() == 1;
    }

    // Safe access
    const UniformDim* DimensionSpec::as_uniform() const
    {
        return boost::get<UniformDim>(&spec_);
    }

    const JaggedDim* DimensionSpec::as_jagged() const
    {
        return boost::get<JaggedDim>(&spec_);
    }

    // Convenient getters (with error checking)
    Index DimensionSpec::uniform_size() const
    {
        const UniformDim* u = as_uniform();
        if (!u)
        {
            throw std::logic_error("uniform_size() is only valid for uniform dimensions");
        }
        return u->size;
    }

    const std::vector<std::size_t>& DimensionSpec::jagged_sizes() const
    {
        const JaggedDim* j = as_jagged();
        if (!j)
        {
            throw std::logic_error("jagged_sizes() is only valid for jagged dimensions");
        }
        return j->sizes;
    }

    const std::vector<std::size_t>& DimensionSpec::prefix_sum() const
    {
        const JaggedDim* j = as_jagged();
        if (j)
        {
            return j->prefix_sum;
        }
        static const std::vector<std::size_t> empty_vec;
        return empty_vec;
    }

    std::size_t DimensionSpec::element_count() const
    {
        if (is_uniform())
        {
            return static_cast<std::size_t>(as_uniform()->size);
        }
        else
        {
            return as_jagged()->sizes.size();
        }
    }

    std::size_t DimensionSpec::child_width(std::size_t parent_idx) const
    {
        if (is_uniform())
        {
            return static_cast<std::size_t>(as_uniform()->size);
        }
        else
        {
            const JaggedDim* j = as_jagged();
            if (parent_idx >= j->sizes.size())
            {
                throw std::out_of_range("parent_idx out of bounds");
            }
            return static_cast<std::size_t>(j->sizes[parent_idx]);
        }
    }

    void DimensionSpec::child_range(std::size_t parent_idx, std::size_t& start, std::size_t& end) const
    {
        const JaggedDim* jagged = as_jagged();
        if (!jagged)
        {
            throw std::logic_error("child_range() is only valid for jagged dimensions");
        }
        if (parent_idx >= jagged->sizes.size())
        {
            throw std::out_of_range("parent_idx out of bounds");
        }
        start = jagged->prefix_sum[parent_idx];
        end = jagged->prefix_sum[parent_idx + 1];
    }

} // namespace xdataset
