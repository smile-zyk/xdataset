#ifndef XDATASET_VARIABLE_DATA_H_
#define XDATASET_VARIABLE_DATA_H_

#include "multi_dimension.h"
#include "cell_series.h"

#include <stdexcept>
#include <vector>

namespace xdataset {

// Thin wrapper around CellSeries that delegates all indexing work to MultiDimensionSpec.
class VariableData {
public:
    VariableData() {}

    VariableData(const MultiDimensionSpec& spec, const CellSeries& series)
        : spec_(spec), series_(series) {
        validate_shape();
    }

    const MultiDimensionSpec& multi_dimension_spec() const { return spec_; }

    const CellSeries& series() const { return series_; }
    CellSeries& series() { return series_; }

    std::size_t size() const { return series_.size(); }

    std::size_t flat_index(const std::vector<Index>& exact_index) const {
        return spec_.flat_index(exact_index);
    }

    std::vector<Index> multi_index(std::size_t flat) const {
        return spec_.multi_index(flat);
    }

    std::vector<std::size_t> flat_indices(const std::vector<MultiIndexSelector>& selectors) const {
        return spec_.flat_indices(selectors);
    }

    std::vector<std::size_t> flat_indices(const std::vector<Index>& selectors_with_any) const {
        return spec_.flat_indices(selectors_with_any);
    }

private:
    void validate_shape() const {
        if (spec_.total_cell_count() != series_.size()) {
            throw std::invalid_argument(
                "multi-dimension total cell count must equal series size");
        }
    }

    MultiDimensionSpec spec_;
    CellSeries series_;
};

}  // namespace xdataset

#endif  // XDATASET_VARIABLE_DATA_H_
