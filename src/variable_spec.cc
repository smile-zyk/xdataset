#include "variable_spec.h"
#include <stdexcept>

namespace xdataset
{
    VariableSpec::VariableSpec(const std::string& name)
        : name_(name), data_(), multi_dimension_spec_(), kind_(VariableKind::kDependent)
    {
    }

    VariableSpec::VariableSpec(const VariableSpecCreateInfo& info)
        : name_(info.name), data_(info.data), 
          multi_dimension_spec_(info.multi_dimension_spec), kind_(info.kind)
    {
        validate();
    }

    VariableSpec::VariableSpec(VariableSpecCreateInfo&& info)
        : name_(std::move(info.name)), 
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

        // Rank must be non-zero for all variable specs.
        if (multi_dimension_spec_.rank() == 0)
        {
            throw std::invalid_argument(
                "MultiDimensionSpec rank must be greater than 0");
        }
    }
} // namespace xdataset
