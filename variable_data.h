#ifndef XDATASET_VARIABLE_DATA_H_
#define XDATASET_VARIABLE_DATA_H_

#include "multi_dimension.h"
#include "cell_series.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
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

    VariableData(MultiDimensionSpec&& spec, CellSeries&& series)
        : spec_(std::move(spec)), series_(std::move(series)) {
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

    VariableData select(const std::vector<MultiIndexSelector>& selectors) const {
        const std::vector<std::size_t> retained_dims = retained_dimension_positions(selectors);
        const MultiDimensionSpec::ProjectedSelectionResult selection =
            spec_.select_with_projection(selectors, retained_dims);

        CellSeries selected_series(series_.cell_kind(), series_.dtype(), series_.cell_shape());
        selected_series.resize(selection.flat_indices.size());

        for (std::size_t i = 0; i < selection.flat_indices.size(); ++i) {
            selected_series.assign_from(series_, selection.flat_indices[i], i);
        }

        return VariableData(
            build_projected_spec(selection.projected_multi_indices, retained_dims.size()),
            std::move(selected_series));
    }

private:
    static std::vector<std::size_t> retained_dimension_positions(
        const std::vector<MultiIndexSelector>& selectors) {
        std::vector<std::size_t> retained_dims;
        retained_dims.reserve(selectors.size());
        for (std::size_t d = 0; d < selectors.size(); ++d) {
            if (!selectors[d].is_equal()) {
                retained_dims.push_back(d);
            }
        }
        return retained_dims;
    }

    static MultiDimensionSpec build_projected_spec(
        const std::vector<std::vector<Index> >& projected_multi_indices,
        std::size_t projected_rank) {
        if (projected_multi_indices.empty()) {
            MultiDimensionSpec empty_spec;
            if (projected_rank == 0) {
                empty_spec.add_uniform(0);
                return empty_spec;
            }
            for (std::size_t i = 0; i < projected_rank; ++i) {
                empty_spec.add_uniform(0);
            }
            return empty_spec;
        }

        MultiDimensionSpec out;
        std::vector<std::pair<std::size_t, std::size_t> > groups;
        groups.push_back(std::make_pair(std::size_t(0), projected_multi_indices.size()));

        for (std::size_t dim = 0; dim < projected_rank; ++dim) {
            std::vector<std::pair<std::size_t, std::size_t> > next_groups;
            std::vector<Index> child_counts;
            child_counts.reserve(groups.size());

            bool first_group = true;
            std::size_t first_child_count = 0;
            bool all_same = true;

            for (std::size_t g = 0; g < groups.size(); ++g) {
                const std::size_t begin = groups[g].first;
                const std::size_t end = groups[g].second;
                std::size_t child_count = 0;

                for (std::size_t i = begin; i < end;) {
                    const Index value = projected_multi_indices[i][dim];
                    std::size_t j = i + 1;
                    while (j < end && projected_multi_indices[j][dim] == value) {
                        ++j;
                    }
                    next_groups.push_back(std::make_pair(i, j));
                    ++child_count;
                    i = j;
                }

                child_counts.push_back(static_cast<Index>(child_count));
                if (first_group) {
                    first_group = false;
                    first_child_count = child_count;
                } else if (child_count != first_child_count) {
                    all_same = false;
                }
            }

            if (all_same) {
                out.add_uniform(static_cast<Index>(first_child_count));
            } else {
                out.add_jagged(child_counts);
            }

            groups.swap(next_groups);
        }

        return out;
    }

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
