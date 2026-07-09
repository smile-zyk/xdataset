#ifndef BLOCK_H
#define BLOCK_H

#include <memory>
#include <string>
#include <vector>
#include <tsl/ordered_map.h>
#include "cell_series.h"
#include "dimension_spec.h"
#include "table_data.h"
#include "variable_data.h"
#include "variable_spec.h"

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
        
        const VariableSpec& variable_spec(const std::string& name) const;

        std::shared_ptr<VariableData> GetOrCreateVariableData(const std::string& variable_name);
        TableData ToTableData();

    private:
        std::shared_ptr<VariableData> CreateVariableData(const VariableSpec& spec);

        std::string name_;
        std::vector<std::string> dependents_;
        std::vector<std::string> independents_;
        tsl::ordered_map<std::string, VariableSpec> variable_spec_map_;
        tsl::ordered_map<std::string, std::shared_ptr<VariableData>> variable_data_cache_;
    };
}

#endif  // BLOCK_H