#ifndef VARIABLE_H
#define VARIABLE_H

#include <memory>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

#include "cell_series.h"
#include "grid_model.h"
#include "multi_dimension_spec.h"
#include "multi_index_selector.h"
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

        const GridModel& grid_model() const;

        const tsl::ordered_map<std::string, CellSeries>& indep_datas() const
        {
            return indep_datas_;
        }

        std::shared_ptr<Variable> indep(Index index = 1) const;

        std::shared_ptr<Variable> indep(const std::string& name) const;

        std::shared_ptr<Variable> at(const std::vector<MultiIndexSelector>& selectors) const;

        std::shared_ptr<Variable> select(const std::vector<MultiIndexSelector>& selectors) const;

        // ── Static factory methods for direct Variable creation ────────────────
        //  All created variables use uniform dimensions.  For jagged or other
        //  complex setups, use Block with IndependentVariableCreateInfo /
        //  DependentVariableCreateInfo instead.

        // Standalone independent variable (no prior independents).
        static std::shared_ptr<Variable> CreateIndependent(
            std::string name,
            CellSeries data);

        // Dependent variable from existing independent Variable objects.
        static std::shared_ptr<Variable> CreateDependent(
            std::string name,
            CellSeries data,
            const std::vector<std::shared_ptr<Variable>>& indep_variables);

    private:
        std::string name_;
        CellSeries data_;
        tsl::ordered_map<std::string, CellSeries> indep_datas_;
        MultiDimensionSpec multi_dimension_spec_;
        VariableKind kind_;
        mutable std::unique_ptr<GridModel> grid_model_cache_;
    };
} // namespace xdataset

#endif // VARIABLE_H
