#ifndef VARIABLE_H
#define VARIABLE_H

#include <memory>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

#include "cell_series.h"
#include "multi_dimension_spec.h"
#include "multi_index_selector.h"
#include "table_data.h"
#include "variable_descriptor.h"

namespace xdataset
{
    struct VariableCreateInfo
    {
        std::string name;
        CellSeries data;
        tsl::ordered_map<std::string, CellSeries> indep_datas;
        MultiDimensionSpec multi_dimension_spec;
        VariableKind kind = VariableKind::kDependent;
    };

    class Variable
    {
    public:
        explicit Variable(const VariableCreateInfo& info);
        explicit Variable(VariableCreateInfo&& info);

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

        const std::string& name() const
        {
            return name_;
        }

        const TableData& GetOrCreateTableData() const;

        std::shared_ptr<Variable> indep(Index index = 1) const;

        std::shared_ptr<Variable> indep(const std::string& name) const;

        std::shared_ptr<Variable> at(const std::vector<MultiIndexSelector>& selectors) const;

        std::shared_ptr<Variable> select(const std::vector<MultiIndexSelector>& selectors) const;

    private:
        std::string name_;
        CellSeries data_;
        tsl::ordered_map<std::string, CellSeries> indep_datas_;
        MultiDimensionSpec multi_dimension_spec_;
        VariableKind kind_;
        mutable bool table_data_cache_valid_ = false;
        mutable TableData table_data_cache_;
    };
} // namespace xdataset

#endif // VARIABLE_H
