#ifndef VARIABLE_SPEC_H
#define VARIABLE_SPEC_H

#include "cell_series.h"
#include "multi_dimension_spec.h"
#include <string>
#include <vector>

namespace xdataset
{
    enum class VariableKind
    {
        kDependent,
        kIndependent
    };

    struct VariableSpecCreateInfo
    {
        std::string name;
        CellSeries data;
        MultiDimensionSpec multi_dimension_spec;
        std::vector<std::string> dependencies;
        VariableKind kind = VariableKind::kDependent;
    };

    class VariableSpec
    {
    public:
        explicit VariableSpec(const std::string& name);
        
        explicit VariableSpec(const VariableSpecCreateInfo& info);
        explicit VariableSpec(VariableSpecCreateInfo&& info);

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

        const std::vector<std::string>& dependencies() const
        {
            return dependencies_;
        }

        VariableKind kind() const
        {
            return kind_;
        }

    private:
        void validate() const;

        std::string name_;
        std::vector<std::string> dependencies_;
        CellSeries data_;
        MultiDimensionSpec multi_dimension_spec_;
        VariableKind kind_;
    };
} // namespace xdataset

#endif // VARIABLE_SPEC_H