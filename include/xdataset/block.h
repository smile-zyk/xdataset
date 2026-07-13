#ifndef BLOCK_H
#define BLOCK_H

#include <memory>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

#include "cell_series.h"
#include "dimension_spec.h"
#include "table_data.h"
#include "variable.h"
#include "variable_descriptor.h"

namespace xdataset
{
    struct IndependentVariableCreateInfo
    {
        std::string name;
        CellSeries data;
        DimensionSpec dimension;
    };

    struct DependentVariableCreateInfo
    {
        std::string name;
        CellSeries data;
    };

    struct BlockCreateInfo
    {
        std::string name;
        std::vector<IndependentVariableCreateInfo> independent_variables;
        std::vector<DependentVariableCreateInfo> dependent_variables;
    };

    class Block
    {
    public:
        explicit Block(const BlockCreateInfo& info);
        explicit Block(BlockCreateInfo&& info);

        const std::string& name() const;
        
        const std::vector<std::string>& dependents() const;
        
        const std::vector<std::string>& independents() const;
        
        const VariableDescriptor& variable_descriptor(const std::string& name) const;

        std::shared_ptr<Variable> GetOrCreateVariable(const std::string& variable_name);
        const TableData& GetOrCreateTableData() const;

    private:
        std::shared_ptr<Variable> CreateVariable(const VariableDescriptor& descriptor);

        std::string name_;
        std::vector<std::string> dependents_;
        std::vector<std::string> independents_;
        tsl::ordered_map<std::string, VariableDescriptor> variable_descriptor_map_;
        tsl::ordered_map<std::string, std::shared_ptr<Variable>> variable_cache_;
        mutable bool table_data_cache_valid_ = false;
        mutable TableData table_data_cache_;
    };
}

#endif  // BLOCK_H