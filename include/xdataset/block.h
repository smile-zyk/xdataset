#ifndef BLOCK_H
#define BLOCK_H

#include <memory>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

#include "cell_series.h"
#include "dimension_spec.h"
#include "grid_model.h"
#include "multi_dimension_spec.h"
#include "variable.h"

namespace xdataset
{
    struct IndependentVariableInfo
    {
        std::string   name;
        CellSeries    data;
        DimensionSpec dimension;
    };

    struct DependentVariableInfo
    {
        std::string name;
        CellSeries  data;
    };

    struct BlockCreateInfo
    {
        std::string                                name;
        std::vector<IndependentVariableInfo> independent_variables;
        std::vector<DependentVariableInfo>   dependent_variables;
    };

    class Block
    {
    public:
        explicit Block(const BlockCreateInfo& info);
        explicit Block(BlockCreateInfo&& info);

        const std::string& name() const;

        std::vector<std::string> dependents() const;

        std::vector<std::string> independents() const;

        const IndependentVariableInfo& independent_variable(const std::string& name) const;

        const DependentVariableInfo& dependent_variable(const std::string& name) const;

        std::shared_ptr<Variable> GetOrCreateVariable(const std::string& variable_name);
        const GridModel& grid_model() const;

    private:
        std::shared_ptr<Variable> CreateVariable(const IndependentVariableInfo& info) const;

        void ensure_unique_name(const std::string& name) const;

        std::string                                           name_;
        tsl::ordered_map<std::string, IndependentVariableInfo> independent_variable_map_;
        tsl::ordered_map<std::string, DependentVariableInfo>   dependent_variable_map_;
        tsl::ordered_map<std::string, std::shared_ptr<Variable>> variable_cache_;
        mutable std::unique_ptr<GridModel>                    grid_model_cache_;
    };
}

#endif  // BLOCK_H