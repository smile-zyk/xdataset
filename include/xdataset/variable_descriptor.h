#ifndef VARIABLE_DESCRIPTOR_H
#define VARIABLE_DESCRIPTOR_H

#include "cell_series.h"
#include "multi_dimension_spec.h"

#include <string>

namespace xdataset
{
    enum class VariableKind
    {
        kDependent,
        kIndependent
    };

    struct VariableDescriptorCreateInfo
    {
        std::string name;
        CellSeries data;
        MultiDimensionSpec multi_dimension_spec;
        VariableKind kind = VariableKind::kDependent;
    };

    class VariableDescriptor
    {
    public:
        explicit VariableDescriptor(const std::string& name);

        explicit VariableDescriptor(const VariableDescriptorCreateInfo& info);
        explicit VariableDescriptor(VariableDescriptorCreateInfo&& info);

        const std::string& name() const
        {
            return name_;
        }

        const CellSeries& data() const
        {
            return data_;
        }

        const MultiDimensionSpec& multi_dimension_spec() const
        {
            return multi_dimension_spec_;
        }

        VariableKind kind() const
        {
            return kind_;
        }

    private:
        void validate() const;

        std::string name_;
        CellSeries data_;
        MultiDimensionSpec multi_dimension_spec_;
        VariableKind kind_;
    };
} // namespace xdataset

#endif // VARIABLE_DESCRIPTOR_H
