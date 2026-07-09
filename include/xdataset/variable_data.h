#ifndef VARIABLE_DATA_H
#define VARIABLE_DATA_H

#include <string>
#include <tsl/ordered_map.h>
#include "cell_series.h"
#include "multi_dimension_spec.h"
#include "table_data.h"
#include "variable_spec.h"

namespace xdataset
{
    struct VariableDataCreateInfo
    {
        std::string name;
        CellSeries data;
        tsl::ordered_map<std::string, CellSeries> indep_datas;
        MultiDimensionSpec multi_dimension_spec;
        VariableKind kind = VariableKind::kDependent;
    };

    class VariableData
    {
    public:
        explicit VariableData(const VariableDataCreateInfo& info);
        explicit VariableData(VariableDataCreateInfo&& info);

        const CellSeries& data() const
        {
            return data_;
        }

        CellSeries& data()
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

        TableData ToTableData() const;

    private:
        std::string name_;
        CellSeries data_;
        tsl::ordered_map<std::string, CellSeries> indep_datas_;
        MultiDimensionSpec multi_dimension_spec_;
        VariableKind kind_;
    };
} // namespace xdataset

#endif // VARIABLE_DATA_H
