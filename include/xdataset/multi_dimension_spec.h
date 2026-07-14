#ifndef MULTI_DIMENSION_SPEC_H
#define MULTI_DIMENSION_SPEC_H

#include "dimension_spec.h"
#include <functional>
#include <vector>

namespace xdataset
{
    class MultiDimensionSpec
    {
    public:
        struct LeafRow
        {
            std::vector<std::size_t> multi_index;
            std::vector<std::size_t> dimension_row_indices;
            std::size_t row_flat = 0;
        };

        using LeafRowVisitor = std::function<void(const LeafRow& leaf_row)>;

        MultiDimensionSpec();
        
        explicit MultiDimensionSpec(const std::vector<DimensionSpec>& dims);

        MultiDimensionSpec& add_uniform(std::size_t size);
        
        MultiDimensionSpec& add_jagged(const std::vector<std::size_t>& sizes);
        
        MultiDimensionSpec& add_dimension(const DimensionSpec& dim);

        std::size_t rank() const;
        
        std::size_t ndim() const;
        
        bool empty() const;
        
        const std::vector<DimensionSpec>& dims() const;

        const DimensionSpec& dim(std::size_t index) const;

        void set_dimensions(const std::vector<DimensionSpec>& dims);

        // CSR-based index conversion: multi-index to flat index
        std::size_t flat_index(const std::vector<std::size_t>& multi_index) const;
        
        // CSR-based reverse: flat index to multi-index
        std::vector<std::size_t> multi_index(std::size_t flat) const;

        // Compute total cell count based on current dimensions
        std::size_t compute_cell_count() const;

        // Visit each leaf row in row-major order and provide per-dimension source row positions.
        void for_each_leaf_row(const LeafRowVisitor& visitor) const;

    private:
        std::size_t compute_cell_count_from(std::size_t start_dim) const;
        std::vector<DimensionSpec> dims_;
    };
} // namespace xdataset

#endif // MULTI_DIMENSION_SPEC_H
