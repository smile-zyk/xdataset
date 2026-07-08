#include "variable_spec.h"
#include <stdexcept>

namespace xdataset
{
    VariableSpec::VariableSpec(const std::string& name)
        : name_(name), dependencies_(), data_(), multi_dimension_spec_(), kind_(VariableKind::kDependent)
    {
    }

    VariableSpec::VariableSpec(const VariableSpecCreateInfo& info)
        : name_(info.name), dependencies_(info.dependencies), data_(info.data), 
          multi_dimension_spec_(info.multi_dimension_spec), kind_(info.kind)
    {
        validate();
    }

    VariableSpec::VariableSpec(VariableSpecCreateInfo&& info)
        : name_(std::move(info.name)), dependencies_(std::move(info.dependencies)), 
          data_(std::move(info.data)), multi_dimension_spec_(std::move(info.multi_dimension_spec)), kind_(info.kind)
    {
        validate();
    }

    void VariableSpec::validate() const
    {
        // Validate shape: if both data and spec are non-empty, they must be consistent
        if (data_.size() > 0 && !multi_dimension_spec_.empty())
        {
            std::size_t expected_count = multi_dimension_spec_.compute_cell_count();
            if (data_.size() != expected_count)
            {
                throw std::invalid_argument(
                    "CellSeries size must match MultiDimensionSpec cell count");
            }
        }

        // Validate dependencies: if rank == 1, must be empty; otherwise must match rank
        std::size_t rank = multi_dimension_spec_.rank();
        if (rank == 1)
        {
            if (!dependencies_.empty())
            {
                throw std::invalid_argument(
                    "dependencies must be empty when MultiDimensionSpec rank is 1");
            }
        }
        else if (dependencies_.size() != rank)
        {
            throw std::invalid_argument(
                "dependencies count must match MultiDimensionSpec rank");
        }
    }
} // namespace xdataset
