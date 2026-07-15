#include "variable.h"
#include "cell_series.h"
#include "dimension_spec.h"
#include "multi_dimension_spec.h"
#include "variable_descriptor.h"

#include <functional>
#include <map>
#include <stdexcept>
#include <tsl/ordered_map.h>
#include <vector>

namespace xdataset
{
    namespace
    {
        CellSeries MakeIndexSeries(std::size_t size)
        {
            CellSeries index_series = CellSeries::Scalars<int>(size, 0);
            for (std::size_t i = 0; i < size; ++i)
            {
                index_series.scalar_at<int>(i) = static_cast<int>(i);
            }
            return index_series;
        }
    } // namespace

    Variable::Variable(const VariableCreateInfo& info)
        : name_(info.name),
          data_(info.data),
          indep_datas_(info.indep_datas),
          multi_dimension_spec_(info.multi_dimension_spec),
          kind_(info.kind)
    {
    }

    Variable::Variable(VariableCreateInfo&& info)
        : name_(std::move(info.name)),
          data_(std::move(info.data)),
          indep_datas_(std::move(info.indep_datas)),
          multi_dimension_spec_(std::move(info.multi_dimension_spec)),
          kind_(info.kind)
    {
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
        {
            throw std::invalid_argument("indep index must be 1-based and greater than 0");
        }

        const std::size_t indep_count = indep_datas_.size();
        const bool has_self_dimension = (kind_ == VariableKind::kIndependent);
        const std::size_t combined_count = indep_count + (has_self_dimension ? 1u : 0u);

        if (static_cast<std::size_t>(index) > combined_count)
        {
            throw std::out_of_range("indep index out of range");
        }

        const std::size_t target_pos = combined_count - static_cast<std::size_t>(index);
        const bool target_is_self = has_self_dimension && (target_pos == indep_count);

        VariableCreateInfo info;
        info.name = target_is_self ? name_ : std::string();
        info.kind = VariableKind::kIndependent;

        std::vector<DimensionSpec> prefix_dims;
        const std::vector<DimensionSpec>& dims = multi_dimension_spec_.dims();

        if (target_is_self)
        {
            info.data = MakeIndexSeries(data_.size());
            info.indep_datas = indep_datas_;
            prefix_dims = dims;
        }
        else
        {
            std::size_t pos = 0;
            for (const auto& item : indep_datas_)
            {
                if (pos < target_pos)
                {
                    info.indep_datas.emplace(item.first, item.second);
                }
                else if (pos == target_pos)
                {
                    info.name = item.first;
                    info.data = item.second;
                }
                ++pos;
            }

            if (info.name.empty())
            {
                throw std::logic_error("failed to resolve target independent variable");
            }

            const std::size_t dim_keep = target_pos + 1;
            if (dims.size() < dim_keep)
            {
                throw std::logic_error("multi-dimension spec rank is smaller than expected");
            }
            prefix_dims.assign(dims.begin(), dims.begin() + static_cast<std::ptrdiff_t>(dim_keep));
        }

        info.multi_dimension_spec = MultiDimensionSpec(prefix_dims);
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
                if (kind_ == VariableKind::kDependent)
                {
                    const std::size_t index_1_based = indep_count - pos;
                    return indep(static_cast<Index>(index_1_based));
                }

                const std::size_t index_1_based = (indep_count + 1) - pos;
                return indep(static_cast<Index>(index_1_based));
            }
            ++pos;
        }

        throw std::invalid_argument("indep name not found: " + name);
    }

    std::shared_ptr<Variable> Variable::at(const std::vector<MultiIndexSelector>& selectors) const
    {
        if (data_.cell_kind() == CellKind::kScalar)
        {
            throw std::logic_error("at is invalid for scalar data");
        }

        VariableCreateInfo info;
        info.name = name_;
        info.kind = kind_;
        info.indep_datas = indep_datas_;
        info.multi_dimension_spec = multi_dimension_spec_;

        if (data_.cell_kind() == CellKind::kVector)
        {
            if (selectors.size() != 1)
            {
                throw std::invalid_argument("vector at requires exactly one selector");
            }

            const std::vector<Index> selected = selectors[0].resolve(data_.cell_shape()[0]);
            info.data = data_.at(selected, selectors[0].is_equal());

            return std::make_shared<Variable>(std::move(info));
        }

        if (selectors.size() != 2)
        {
            throw std::invalid_argument("matrix at requires exactly two selectors");
        }

        const std::vector<Index> selected_rows = selectors[0].resolve(data_.cell_shape()[0]);
        const std::vector<Index> selected_cols = selectors[1].resolve(data_.cell_shape()[1]);
        info.data = data_.at(
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

        const std::size_t self_dim = rank - 1;
        std::vector<bool> is_dim_retain(rank, true);
        for (std::size_t source_dim = 0; source_dim < rank; ++source_dim)
        {
            const bool is_equal_selector = actual_selectors[source_dim].is_equal();
            // For an Independent variable, the last dimension represents its own data column,
            // so we keep that dimension even when it is selected by equality.
            const bool is_independent_self =
                (kind_ == VariableKind::kIndependent) && (source_dim == self_dim);
            is_dim_retain[source_dim] = !is_equal_selector || is_independent_self;
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
            if (dim_idx == rank)
            {
                selected_row_indices.push_back(parent_flat);
                return;
            }

            const DimensionSpec& dim = multi_dimension_spec_.dim(dim_idx);
            std::size_t width = 0;
            if (dim.is_uniform())
            {
                width = static_cast<std::size_t>(dim.uniform_size());
            }
            else
            {
                width = dim.child_width(parent_flat);
            }

            const std::vector<Index> selected_children =
                actual_selectors[dim_idx].resolve(static_cast<Index>(width));

            selection_info[dim_idx].child_counts.push_back(
                static_cast<std::size_t>(selected_children.size()));
            selection_info[dim_idx].is_jagged = !dim.is_uniform();

            if (selection_info[dim_idx].is_jagged)
            {
                for (Index child : selected_children)
                {
                    std::size_t start = 0;
                    std::size_t end = 0;
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

            for (std::size_t child : selected_children)
            {
                std::size_t current_flat = 0;
                if (dim.is_uniform())
                {
                    const std::size_t size = static_cast<std::size_t>(dim.uniform_size());
                    current_flat = parent_flat * size + child;
                }
                else
                {
                    std::size_t start = 0;
                    std::size_t end = 0;
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

            const std::vector<std::size_t>& counts = selection_info[i].child_counts;
            if (counts.empty())
            {
                selected_multi_dim.add_uniform(0);
                continue;
            }

            if (selection_info[i].is_jagged)
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
        CellSeries selected_data = CellSeries(data_.cell_kind(), data_.dtype(), data_.cell_shape());
        if (kind_ == VariableKind::kDependent)
        {
            for (const Index& source_row : selected_row_indices)
            {
                selected_data.append_from(data_, source_row);
            }
            info.data = std::move(selected_data);
        }
        else if (kind_ == VariableKind::kIndependent)
        {
            const auto& source_rows = selection_info[self_dim].source_rows;
            for (std::size_t source_row : source_rows)
            {
                selected_data.append_from(data_, source_row);
            }
            info.data = std::move(selected_data);
        }

        for (std::size_t source_dim_idx = 0; source_dim_idx < rank; ++source_dim_idx)
        {
            if (!is_dim_retain[source_dim_idx])
            {
                continue;
            }

            const bool is_independent_self =
                (kind_ == VariableKind::kIndependent) && (source_dim_idx == self_dim);

            if (is_independent_self)
            {
                continue;
            }

            const auto& source_rows = selection_info[source_dim_idx].source_rows;

            const std::string& indep_name = std::next(indep_datas_.begin(), source_dim_idx)->first;
            const CellSeries& indep_series = indep_datas_.at(indep_name);
            for (std::size_t source_row : source_rows)
            {
                info.indep_datas[indep_name].append_from(indep_series, source_row);
            }
        }

        return std::make_shared<Variable>(std::move(info));
    }

    // ── Static factory methods ─────────────────────────────────────────────────

    std::shared_ptr<Variable> Variable::CreateIndependent(
        std::string name,
        CellSeries data)
    {
        const Index size = static_cast<Index>(data.size());
        VariableCreateInfo vinfo;
        vinfo.name = std::move(name);
        vinfo.data = std::move(data);
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
        std::vector<DimensionSpec> indep_dimensions;

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
            indep_dimensions.push_back(dims.back());
        }

        VariableCreateInfo vinfo;
        vinfo.name = std::move(name);
        vinfo.data = std::move(data);
        vinfo.indep_datas = std::move(indep_datas);
        vinfo.multi_dimension_spec = MultiDimensionSpec(std::move(indep_dimensions));
        vinfo.kind = VariableKind::kDependent;
        return std::make_shared<Variable>(std::move(vinfo));
    }
} // namespace xdataset
