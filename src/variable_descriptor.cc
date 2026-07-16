#include "variable_descriptor.h"

#include <stdexcept>

namespace xdataset
{
    VariableDescriptor::VariableDescriptor(const std::string& name)
        : name_(name), data_(), multi_dimension_spec_(), kind_(VariableKind::kDependent)
    {
    }

    VariableDescriptor::VariableDescriptor(const VariableDescriptorCreateInfo& info)
        : name_(info.name),
          data_(info.data),
          multi_dimension_spec_(info.multi_dimension_spec),
          kind_(info.kind)
    {
        validate();
    }

    VariableDescriptor::VariableDescriptor(VariableDescriptorCreateInfo&& info)
        : name_(std::move(info.name)),
          data_(std::move(info.data)),
          multi_dimension_spec_(std::move(info.multi_dimension_spec)),
          kind_(info.kind)
    {
        validate();
    }

    void VariableDescriptor::validate() const
    {
        if (data_.size() > 0 && !multi_dimension_spec_.empty())
        {
            std::size_t expected_count = multi_dimension_spec_.compute_cell_count();
            if (data_.size() != expected_count)
            {
                throw std::invalid_argument(
                    "CellSeries size must match MultiDimensionSpec cell count");
            }
        }

        if (multi_dimension_spec_.rank() == 0)
        {
            throw std::invalid_argument(
                "MultiDimensionSpec rank must be greater than 0");
        }
    }
} // namespace xdataset
