#ifndef MULTI_DIMENSION_SPEC_H
#define MULTI_DIMENSION_SPEC_H

#include "dimension_spec.h"
#include "multi_index_selector.h"
#include <vector>

namespace xdataset
{
    class MultiDimensionSpec
    {
    public:
        MultiDimensionSpec();
        
        explicit MultiDimensionSpec(const std::vector<DimensionSpec>& dims);

        MultiDimensionSpec& add_uniform(Index size);
        
        MultiDimensionSpec& add_jagged(const std::vector<Index>& sizes);
        
        MultiDimensionSpec& add_dimension(const DimensionSpec& dim);

        std::size_t rank() const;
        
        std::size_t ndim() const;
        
        bool empty() const;
        
        const std::vector<DimensionSpec>& dims() const;

        void set_dimensions(const std::vector<DimensionSpec>& dims);

        // CSR-based index conversion: multi-index to flat index
        std::size_t flat_index(const std::vector<std::size_t>& multi_index) const;
        
        // CSR-based reverse: flat index to multi-index
        std::vector<std::size_t> multi_index(std::size_t flat) const;

        // Compute total cell count based on current dimensions
        std::size_t compute_cell_count() const;

        // Select dimensions based on selectors: Any retains dimension, Equal drops it, In converts to jagged
        MultiDimensionSpec select(const std::vector<MultiIndexSelector>& selectors) const;

    private:
        std::size_t compute_cell_count_from(std::size_t start_dim) const;
        std::vector<DimensionSpec> dims_;
    };
} // namespace xdataset

#endif // MULTI_DIMENSION_SPEC_H
