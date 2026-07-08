#ifndef VARIABLE_DATA_H
#define VARIABLE_DATA_H

#include <memory>
#include <tsl/ordered_map.h>
#include "cell_series.h"
#include "variable_spec.h"
#include "multi_dimension_spec.h"
#include "multi_index_selector.h"

namespace xdataset
{
    class Block;
    class VariableData
    {
    public:
        VariableData(const CellSeries& data = CellSeries());
        VariableData(CellSeries&& data = CellSeries());

        ~VariableData();

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

        std::shared_ptr<VariableData> Clone() const;
        void ClearBlockReference();

    private:
        friend class Block;
        void SetBlockReference(const std::shared_ptr<Block>& block);

        CellSeries data_;
        tsl::ordered_map<std::string, CellSeries> indep_datas_;
        MultiDimensionSpec multi_dimension_spec_;
        VariableKind kind_;
        std::weak_ptr<Block> block_;
    };
} // namespace xdataset

#endif // VARIABLE_DATA_H
