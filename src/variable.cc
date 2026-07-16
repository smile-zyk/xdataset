#include "variable.h"
#include "cell_series.h"
#include "dimension_spec.h"
#include "multi_dimension_spec.h"

#include <functional>
#include <map>
#include <stdexcept>
#include <tsl/ordered_map.h>
#include <vector>

namespace xdataset
{

    Variable::Variable(const VariableCreateInfo& info)
        : name_(info.name),
          data_(info.data),
          indep_datas_(info.indep_datas),
          multi_dimension_spec_(info.multi_dimension_spec),
          kind_(info.kind)
    {
        if (!multi_dimension_spec_.empty())
        {
            const std::size_t expected = multi_dimension_spec_.compute_cell_count();
            if (data_.size() != static_cast<Index>(expected))
            {
                throw std::invalid_argument(
                    "Variable '" + name_ + "': data size " + std::to_string(data_.size()) +
                    " does not match multi_dimension_spec cell count " + std::to_string(expected));
            }
        }
    }

    Variable::Variable(VariableCreateInfo&& info)
        : name_(std::move(info.name)),
          data_(std::move(info.data)),
          indep_datas_(std::move(info.indep_datas)),
          multi_dimension_spec_(std::move(info.multi_dimension_spec)),
          kind_(info.kind)
    {
        if (!multi_dimension_spec_.empty())
        {
            const std::size_t expected = multi_dimension_spec_.compute_cell_count();
            if (data_.size() != static_cast<Index>(expected))
            {
                throw std::invalid_argument(
                    "Variable '" + name_ + "': data size " + std::to_string(data_.size()) +
                    " does not match multi_dimension_spec cell count " + std::to_string(expected));
            }
        }
    }

    void Variable::set_name(const std::string& name)
    {
        if (kind_ == VariableKind::kIndependent && !indep_datas_.empty() && name != name_)
        {
            auto last = --indep_datas_.end();
            CellSeries col = std::move(last->second);
            indep_datas_.erase(last);
            indep_datas_.emplace(name, std::move(col));
        }
        name_ = name;
    }

    const GridModel& Variable::grid_model() const
    {
        if (!grid_model_cache_)
        {
            grid_model_cache_.reset(new VariableGridModel(*this));
        }
        return *grid_model_cache_;
    }

    std::shared_ptr<Variable> Variable::indep(Index index) const
    {
        if (index <= 0)
            throw std::invalid_argument("indep index must be 1-based and greater than 0");

        const std::size_t count = indep_datas_.size();
        if (static_cast<std::size_t>(index) > count)
            throw std::out_of_range("indep index out of range");

        // index=1 → last column, index=count → first column
        const std::size_t target = count - static_cast<std::size_t>(index);
        const bool        is_self = (kind_ == VariableKind::kIndependent) && (target == count - 1);

        // Build prefix dimensions first (needed for compute_cell_count).
        std::vector<DimensionSpec> prefix_dims;
        VariableCreateInfo         info;
        info.kind = VariableKind::kIndependent;

        std::size_t pos = 0;
        auto it = indep_datas_.begin();
        for (const auto& item : indep_datas_)
        {
            if (pos < target)
            {
                info.indep_datas.emplace(item.first, item.second);
                prefix_dims.push_back(multi_dimension_spec_.dims()[pos]);
            }
            else if (pos == target)
            {
                info.name = item.first;
                it = std::next(indep_datas_.begin(), static_cast<std::ptrdiff_t>(pos));
                prefix_dims.push_back(multi_dimension_spec_.dims()[pos]);
            }
            ++pos;
        }

        const MultiDimensionSpec result_spec(prefix_dims);
        const std::size_t total_rows = result_spec.compute_cell_count();

        // Generate index series for the target column.
        if (is_self)
        {
            // Self: index = multi_index.back() for each leaf row.
            CellSeries idx = CellSeries::Scalars<int>(total_rows, 0);
            result_spec.for_each_leaf_row(
                [&](const MultiDimensionSpec::LeafRow& lr)
                {
                    idx.scalar_at<int>(lr.row_flat) =
                        static_cast<int>(lr.multi_index.back());
                });
            info.data = std::move(idx);
        }
        else
        {
            // Expand raw data to the full cartesian product of result_spec.
            const CellSeries& raw = it->second;
            CellSeries expanded = CellSeries(raw.cell_kind(), raw.dtype(), raw.cell_shape());
            result_spec.for_each_leaf_row(
                [&](const MultiDimensionSpec::LeafRow& lr)
                {
                    expanded.append_from(raw, lr.dimension_row_indices.back());
                });
            info.data = std::move(expanded);
        }
        info.indep_datas[info.name] = info.data;

        info.multi_dimension_spec = result_spec;
        return std::make_shared<Variable>(std::move(info));
    }

    std::shared_ptr<Variable> Variable::indep(const std::string& name) const
    {
        if (name.empty())
        {
            throw std::invalid_argument("indep name must not be empty");
        }

        if (kind_ == VariableKind::kIndependent && name == name_)
        {
            return indep(1);
        }

        std::size_t pos = 0;
        for (const auto& item : indep_datas_)
        {
            if (item.first == name)
            {
                const std::size_t indep_count = indep_datas_.size();
                const Index index_1_based = static_cast<Index>(indep_count - pos);
                return indep(index_1_based);
            }
            ++pos;
        }

        throw std::invalid_argument("indep name not found: " + name);
    }

    std::shared_ptr<Variable> Variable::at(const std::vector<MultiIndexSelector>& selectors) const
    {
        if (data().cell_kind() == CellKind::kScalar)
        {
            throw std::logic_error("at is invalid for scalar data");
        }

        VariableCreateInfo info;
        info.name = name_;
        info.kind = kind_;
        info.indep_datas = indep_datas_;
        info.multi_dimension_spec = multi_dimension_spec_;

        if (data().cell_kind() == CellKind::kVector)
        {
            if (selectors.size() != 1)
            {
                throw std::invalid_argument("vector at requires exactly one selector");
            }

            const std::vector<Index> selected = selectors[0].resolve(data().cell_shape()[0]);
            info.data = data().at(selected, selectors[0].is_equal());

            return std::make_shared<Variable>(std::move(info));
        }

        if (selectors.size() != 2)
        {
            throw std::invalid_argument("matrix at requires exactly two selectors");
        }

        const std::vector<Index> selected_rows = selectors[0].resolve(data().cell_shape()[0]);
        const std::vector<Index> selected_cols = selectors[1].resolve(data().cell_shape()[1]);
        info.data = data().at(
            selected_rows,
            selected_cols,
            selectors[0].is_equal(),
            selectors[1].is_equal());

        return std::make_shared<Variable>(std::move(info));
    }

    std::shared_ptr<Variable> Variable::select(
        const std::vector<MultiIndexSelector>& selectors) const
    {
        const std::size_t rank = multi_dimension_spec_.rank();
        if (rank == 0)
        {
            throw std::logic_error("select requires non-empty dimensions");
        }

        if (selectors.size() > rank)
        {
            throw std::invalid_argument("selector count exceeds variable rank");
        }

        // `selectors` may be shorter than rank: normalize it by prepending Any and
        // keeping the user-provided selectors at the end, so short selectors only
        // constrain the trailing dimensions by default.
        std::vector<MultiIndexSelector> actual_selectors;
        actual_selectors.reserve(rank);
        if (selectors.size() < rank)
        {
            actual_selectors.insert(
                actual_selectors.end(), rank - selectors.size(), MultiIndexSelector::Any());
        }
        actual_selectors.insert(actual_selectors.end(), selectors.begin(), selectors.end());

        std::vector<bool> is_dim_retain(rank, true);
        for (std::size_t source_dim = 0; source_dim < rank; ++source_dim)
        {
            is_dim_retain[source_dim] = !actual_selectors[source_dim].is_equal();
        }

        struct SelectionDimensionInformation
        {
            bool is_jagged = false;
            std::vector<Index> source_rows;
            std::vector<std::size_t> child_counts;
        };

        // key is source dimension index, value is SelectionDimensionInformation
        std::map<Index, SelectionDimensionInformation> selection_info;

        // for dependent variable, record source data selected row
        std::vector<Index> selected_row_indices;

        std::function<void(Index, Index)> walk = [&](Index dim_idx, Index parent_flat)
        {
            if (dim_idx == static_cast<Index>(rank))
            {
                selected_row_indices.push_back(parent_flat);
                return;
            }

            const DimensionSpec& dim = multi_dimension_spec_.dim(dim_idx);
            std::size_t width = 0;
            if (dim.is_uniform())
            {
                width = dim.uniform_size();
            }
            else
            {
                width = dim.child_width(parent_flat);
            }

            const std::vector<Index> selected_children =
                actual_selectors[static_cast<std::size_t>(dim_idx)].resolve(static_cast<Index>(width));

            selection_info[dim_idx].child_counts.push_back(selected_children.size());
            selection_info[dim_idx].is_jagged = !dim.is_uniform();

            if (selection_info[dim_idx].is_jagged)
            {
                for (Index child : selected_children)
                {
                    Index start = 0;
                    Index end = 0;
                    dim.child_range(parent_flat, start, end);
                    Index source_row = start + child;
                    selection_info[dim_idx].source_rows.push_back(source_row);
                }
            }
            else
            {
                if (selection_info[dim_idx].source_rows.empty())
                {
                    selection_info[dim_idx].source_rows = selected_children;
                }
            }

            for (Index child : selected_children)
            {
                Index current_flat = 0;
                if (dim.is_uniform())
                {
                    const Index size = static_cast<Index>(dim.uniform_size());
                    current_flat = parent_flat * size + child;
                }
                else
                {
                    Index start = 0;
                    Index end = 0;
                    dim.child_range(parent_flat, start, end);
                    (void)end;
                    current_flat = start + child;
                }

                walk(dim_idx + 1, current_flat);
            }
        };

        walk(0, 0);

        MultiDimensionSpec selected_multi_dim;
        for (std::size_t i = 0; i < rank; ++i)
        {
            if (!is_dim_retain[i])
            {
                continue;
            }

            const std::vector<std::size_t>& counts = selection_info[static_cast<Index>(i)].child_counts;
            if (counts.empty())
            {
                selected_multi_dim.add_uniform(0);
                continue;
            }

            if (selection_info[static_cast<Index>(i)].is_jagged)
            {
                if (counts.size() == 1)
                {
                    selected_multi_dim.add_uniform(counts.front());
                }
                else
                {
                    selected_multi_dim.add_jagged(counts);
                }
            }
            else
            {
                selected_multi_dim.add_uniform(counts.front());
            }
        }

        VariableCreateInfo info;
        info.name = name_;
        info.kind = kind_;
        info.multi_dimension_spec = selected_multi_dim;
        CellSeries selected_data = CellSeries(data().cell_kind(), data().dtype(), data().cell_shape());
        for (const Index& source_row : selected_row_indices)
        {
            selected_data.append_from(data(), source_row);
        }
        info.data = std::move(selected_data);

        for (std::size_t source_dim_idx = 0; source_dim_idx < rank; ++source_dim_idx)
        {
            if (!is_dim_retain[source_dim_idx])
            {
                continue;
            }

            const auto& source_rows = selection_info[static_cast<Index>(source_dim_idx)].source_rows;

            const std::string& indep_name = std::next(indep_datas_.begin(), source_dim_idx)->first;
            const CellSeries& indep_series = indep_datas_.at(indep_name);
            for (Index source_row : source_rows)
            {
                info.indep_datas[indep_name].append_from(indep_series, source_row);
            }
        }

        if (kind_ == VariableKind::kIndependent)
        {
            info.indep_datas[info.name] = info.data;
        }

        return std::make_shared<Variable>(std::move(info));
    }

    // ── Static factory methods ─────────────────────────────────────────────────

    std::shared_ptr<Variable> Variable::CreateIndependent(
        std::string name,
        CellSeries data)
    {
        const std::size_t size = data.size();
        VariableCreateInfo vinfo;
        vinfo.name = name;
        vinfo.indep_datas[vinfo.name] = data;    // raw copy
        vinfo.data = std::move(data);            // expanded = self for single-dim
        vinfo.multi_dimension_spec = MultiDimensionSpec({DimensionSpec::Uniform(size)});
        vinfo.kind = VariableKind::kIndependent;
        return std::make_shared<Variable>(std::move(vinfo));
    }

    std::shared_ptr<Variable> Variable::CreateDependent(
        std::string name,
        CellSeries data,
        const std::vector<std::shared_ptr<Variable>>& indep_variables)
    {
        if (indep_variables.empty())
        {
            throw std::invalid_argument(
                "CreateDependent: indep_variables must not be empty");
        }

        tsl::ordered_map<std::string, CellSeries> indep_datas;
        MultiDimensionSpec spec;

        for (const auto& var : indep_variables)
        {
            if (!var)
            {
                throw std::invalid_argument(
                    "CreateDependent: null indep_variable in list");
            }
            if (var->kind() != VariableKind::kIndependent)
            {
                throw std::invalid_argument(
                    "CreateDependent: variable '" + var->name() +
                    "' is not an independent variable");
            }

            indep_datas[var->name()] = var->data();

            const std::vector<DimensionSpec>& dims = var->multi_dimension_spec().dims();
            if (dims.empty())
            {
                throw std::logic_error(
                    "CreateDependent: independent variable '" + var->name() +
                    "' has no dimensions");
            }
            spec.add_dimension(dims.back());   // validates parent count for jagged
        }

        VariableCreateInfo vinfo;
        vinfo.name = std::move(name);
        vinfo.data = std::move(data);
        vinfo.indep_datas = std::move(indep_datas);
        vinfo.multi_dimension_spec = std::move(spec);
        vinfo.kind = VariableKind::kDependent;
        return std::make_shared<Variable>(std::move(vinfo));
    }
} // namespace xdataset
