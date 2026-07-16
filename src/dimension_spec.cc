#include "dimension_spec.h"
#include <cstddef>
#include <stdexcept>

namespace xdataset
{
    // Factory methods
    DimensionSpec DimensionSpec::Uniform(std::size_t size)
    {
        return DimensionSpec(UniformDim(size));
    }

    DimensionSpec DimensionSpec::Jagged(const std::vector<std::size_t>& sizes)
    {
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
    std::size_t DimensionSpec::uniform_size() const
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

    std::size_t DimensionSpec::child_width(Index parent_idx) const
    {
        if (is_uniform())
        {
            return as_uniform()->size;
        }
        else
        {
            const JaggedDim* j = as_jagged();
            if (parent_idx < 0 || static_cast<std::size_t>(parent_idx) >= j->sizes.size())
            {
                throw std::out_of_range("parent_idx out of bounds");
            }
            return j->sizes[static_cast<std::size_t>(parent_idx)];
        }
    }

    void DimensionSpec::child_range(Index parent_idx, Index& start, Index& end) const
    {
        const JaggedDim* jagged = as_jagged();
        if (!jagged)
        {
            throw std::logic_error("child_range() is only valid for jagged dimensions");
        }
        if (parent_idx < 0 || static_cast<std::size_t>(parent_idx) >= jagged->sizes.size())
        {
            throw std::out_of_range("parent_idx out of bounds");
        }
        start = static_cast<Index>(jagged->prefix_sum[static_cast<std::size_t>(parent_idx)]);
        end = static_cast<Index>(jagged->prefix_sum[static_cast<std::size_t>(parent_idx) + 1]);
    }

} // namespace xdataset
