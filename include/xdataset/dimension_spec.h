#ifndef DIMENSION_SPEC_H
#define DIMENSION_SPEC_H

#include "xdataset_predefine.h"
#include <boost/variant.hpp>
#include <cstddef>
#include <vector>

namespace xdataset
{
    struct UniformDim
    {
        std::size_t size;
        
        UniformDim() : size(0) {}
        explicit UniformDim(std::size_t s) : size(s) {}
    };

    struct JaggedDim
    {
        std::vector<std::size_t> sizes;
        std::vector<std::size_t> prefix_sum;
        
        JaggedDim() {}
        explicit JaggedDim(const std::vector<std::size_t>& s) : sizes(s)
        {
            compute_prefix_sum();
        }
        
        void compute_prefix_sum()
        {
            prefix_sum.clear();
            prefix_sum.reserve(sizes.size() + 1);
            prefix_sum.push_back(0);
            for (std::size_t i = 0; i < sizes.size(); ++i)
            {
                prefix_sum.push_back(prefix_sum.back() + static_cast<std::size_t>(sizes[i]));
            }
        }
    };

    class DimensionSpec
    {
    public:
        static DimensionSpec Uniform(std::size_t size);
        static DimensionSpec Jagged(const std::vector<std::size_t>& sizes);

        // Type checking
        bool is_uniform() const;
        bool is_jagged() const;

        // Safe access: returns pointer if type matches, nullptr otherwise
        const UniformDim* as_uniform() const;
        const JaggedDim* as_jagged() const;

        // Convenient getters (throws if wrong type)
        std::size_t uniform_size() const;
        const std::vector<std::size_t>& jagged_sizes() const;
        const std::vector<std::size_t>& prefix_sum() const;
        
        // Number of elements at this level (uniform: size, jagged: sizes.size())
        std::size_t element_count() const;
        
        // Width of child elements for a specific parent (uniform: uniform_size, jagged: jagged_sizes[parent_idx])
        std::size_t child_width(Index parent_idx) const;
        
        // Get child range [start, end) for parent at parent_idx (jagged only)
        void child_range(Index parent_idx, Index& start, Index& end) const;

    private:
        explicit DimensionSpec(const UniformDim& uniform);
        explicit DimensionSpec(const JaggedDim& jagged);

        boost::variant<UniformDim, JaggedDim> spec_;
    };
} // namespace xdataset

#endif // DIMENSION_SPEC_H