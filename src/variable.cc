#include "variable.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
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

        std::vector<std::size_t> BuildSelectedChildIndices(
            const MultiIndexSelector& selector,
            std::size_t width)
        {
            std::vector<std::size_t> selected;

            if (selector.is_any())
            {
                selected.reserve(width);
                for (std::size_t i = 0; i < width; ++i)
                {
                    selected.push_back(i);
                }
                return selected;
            }

            if (selector.is_equal())
            {
                const Index value = selector.equal_value();
                if (value >= 0 && static_cast<std::size_t>(value) < width)
                {
                    selected.push_back(static_cast<std::size_t>(value));
                }
                return selected;
            }

            const std::vector<Index>& values = selector.in_values();
            selected.reserve(values.size());
            for (Index value : values)
            {
                if (value >= 0 && static_cast<std::size_t>(value) < width)
                {
                    selected.push_back(static_cast<std::size_t>(value));
                }
            }
            return selected;
        }

        std::size_t RowCountForDimension(const DimensionSpec& dim)
        {
            if (dim.is_uniform())
            {
                return static_cast<std::size_t>(dim.uniform_size());
            }
            return dim.prefix_sum().empty() ? 0u : dim.prefix_sum().back();
        }

        CellSeries MakeEmptyLike(const CellSeries& series)
        {
            return CellSeries(series.cell_kind(), series.dtype(), series.cell_shape());
        }

        CellSeries HeadOrEmpty(const CellSeries& source, std::size_t count)
        {
            if (count > source.size())
            {
                throw std::out_of_range("selected independent row count exceeds source size");
            }
            return source.iloc(0, count);
        }

    } // namespace

    Variable::Variable(const VariableCreateInfo& info)
        : name_(info.name),
          data_(info.data),
          indep_datas_(info.indep_datas),
          multi_dimension_spec_(info.multi_dimension_spec),
          kind_(info.kind),
          table_data_cache_valid_(false),
          table_data_cache_()
    {
    }

    Variable::Variable(VariableCreateInfo&& info)
        : name_(std::move(info.name)),
          data_(std::move(info.data)),
          indep_datas_(std::move(info.indep_datas)),
          multi_dimension_spec_(std::move(info.multi_dimension_spec)),
          kind_(info.kind),
          table_data_cache_valid_(false),
          table_data_cache_()
    {
    }

    const TableData& Variable::GetOrCreateTableData() const
    {
        if (table_data_cache_valid_)
        {
            return table_data_cache_;
        }

        if (multi_dimension_spec_.empty())
        {
            throw std::logic_error("variable table view requires non-empty dimensions");
        }

        TableData table;

        std::vector<std::pair<std::string, const CellSeries*>> indep_columns;
        indep_columns.reserve(indep_datas_.size() + 1);

        for (const auto& item : indep_datas_)
        {
            indep_columns.push_back(std::make_pair(item.first, &item.second));
            const std::vector<std::string> headers =
                TableData::ExpandHeadersForSeries(item.first, item.second);
            table.metadata.headers.insert(
                table.metadata.headers.end(), headers.begin(), headers.end());
        }

        if (kind_ == VariableKind::kIndependent)
        {
            const std::string self_name = name_.empty() ? "self" : name_;
            indep_columns.push_back(std::make_pair(self_name, &data_));
            const std::vector<std::string> headers =
                TableData::ExpandHeadersForSeries(self_name, data_);
            table.metadata.headers.insert(
                table.metadata.headers.end(), headers.begin(), headers.end());
        }

        if (kind_ == VariableKind::kDependent)
        {
            const std::vector<std::string> headers =
                TableData::ExpandHeadersForSeries("data", data_);
            table.metadata.headers.insert(
                table.metadata.headers.end(), headers.begin(), headers.end());
        }

        const std::size_t rank = multi_dimension_spec_.rank();
        if (indep_columns.size() != rank)
        {
            throw std::logic_error("independent columns count must match MultiDimensionSpec rank");
        }

        multi_dimension_spec_.for_each_leaf_row(
            [&](const MultiDimensionSpec::LeafRow& leaf_row)
        {
            const std::vector<std::size_t>& multi_index = leaf_row.multi_index;
            const std::vector<std::size_t>& dimension_row_indices =
                leaf_row.dimension_row_indices;
            const std::size_t row_flat = leaf_row.row_flat;

            std::vector<std::string> row;
            row.reserve(table.metadata.headers.size());

            for (std::size_t dim = 0; dim < indep_columns.size(); ++dim)
            {
                const CellSeries* indep_series = indep_columns[dim].second;
                const std::size_t src_row = dimension_row_indices[dim];
                if (src_row >= indep_series->size())
                {
                    throw std::out_of_range("expanded independent row index out of bounds");
                }
                const std::vector<std::string> values =
                    TableData::CellAtToColumns(*indep_series, src_row);
                row.insert(row.end(), values.begin(), values.end());
            }

            if (kind_ == VariableKind::kDependent)
            {
                if (row_flat >= data_.size())
                {
                    throw std::out_of_range("dependent data row index out of bounds");
                }
                const std::vector<std::string> values =
                    TableData::CellAtToColumns(data_, row_flat);
                row.insert(row.end(), values.begin(), values.end());
            }

            table.metadata.multi_indices.push_back(multi_index);
            table.rows.push_back(row);
        });

        table_data_cache_ = std::move(table);
        table_data_cache_valid_ = true;
        return table_data_cache_;
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

    // std::shared_ptr<Variable> Variable::at(
    //     const std::vector<MultiIndexSelector>& selectors) const
    // {
    // }

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

        std::vector<MultiIndexSelector> normalized_selectors = selectors;
        while (normalized_selectors.size() < rank)
        {
            normalized_selectors.push_back(MultiIndexSelector::Any());
        }

        const std::size_t self_dim = rank - 1;
        std::vector<bool> keep_dim(rank, true);
        for (std::size_t source_dim = 0; source_dim < rank; ++source_dim)
        {
            const bool is_equal_selector = normalized_selectors[source_dim].is_equal();
            const bool is_independent_self =
                (kind_ == VariableKind::kIndependent) && (source_dim == self_dim);
            keep_dim[source_dim] = !is_equal_selector || is_independent_self;
        }

        std::vector<int> source_dim_to_selected_dim(rank, -1);
        std::size_t selected_rank = 0;
        for (std::size_t source_dim = 0; source_dim < rank; ++source_dim)
        {
            if (keep_dim[source_dim])
            {
                source_dim_to_selected_dim[source_dim] = static_cast<int>(selected_rank);
                ++selected_rank;
            }
        }

        const std::vector<DimensionSpec>& source_dims = multi_dimension_spec_.dims();
        std::vector<std::vector<Index>> selected_dim_child_counts(selected_rank);
        std::vector<std::size_t> selected_data_rows;

        std::function<void(std::size_t, std::size_t)> walk =
            [&](std::size_t dim_idx, std::size_t parent_flat)
        {
            if (dim_idx == rank)
            {
                selected_data_rows.push_back(parent_flat);
                return;
            }

            const DimensionSpec& dim = source_dims[dim_idx];
            std::size_t width = 0;
            if (dim.is_uniform())
            {
                width = static_cast<std::size_t>(dim.uniform_size());
            }
            else
            {
                width = dim.child_width(parent_flat);
            }

            const std::vector<std::size_t> selected_children =
                BuildSelectedChildIndices(normalized_selectors[dim_idx], width);

            const int selected_dim_idx = source_dim_to_selected_dim[dim_idx];
            if (selected_dim_idx >= 0)
            {
                selected_dim_child_counts[static_cast<std::size_t>(selected_dim_idx)].push_back(
                    static_cast<Index>(selected_children.size()));
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

        std::vector<DimensionSpec> selected_dims;
        selected_dims.reserve(selected_rank);
        for (std::size_t i = 0; i < selected_rank; ++i)
        {
            const std::vector<Index>& counts = selected_dim_child_counts[i];
            if (counts.empty())
            {
                selected_dims.push_back(DimensionSpec::Uniform(0));
                continue;
            }

            const Index first = counts.front();
            const bool all_same =
                std::all_of(
                    counts.begin(),
                    counts.end(),
                    [&](Index value)
                {
                    return value == first;
                });

            if (all_same)
            {
                selected_dims.push_back(DimensionSpec::Uniform(first));
                continue;
            }

            selected_dims.push_back(DimensionSpec::Jagged(counts));
        }

        VariableCreateInfo info;
        info.name = name_;
        info.kind = kind_;
        info.multi_dimension_spec = MultiDimensionSpec(selected_dims);

        CellSeries selected_data = MakeEmptyLike(data_);
        for (std::size_t source_row : selected_data_rows)
        {
            selected_data.append_from(data_, source_row);
        }
        info.data = std::move(selected_data);

        std::size_t source_dim_idx = 0;
        for (const auto& item : indep_datas_)
        {
            if (source_dim_idx >= rank)
            {
                throw std::logic_error("independent data count exceeds variable rank");
            }

            if (keep_dim[source_dim_idx])
            {
                const int selected_dim_idx = source_dim_to_selected_dim[source_dim_idx];
                if (selected_dim_idx < 0)
                {
                    throw std::logic_error("kept dimension index mapping is invalid");
                }

                const std::size_t selected_rows =
                    RowCountForDimension(selected_dims[static_cast<std::size_t>(selected_dim_idx)]);
                info.indep_datas.emplace(item.first, HeadOrEmpty(item.second, selected_rows));
            }

            ++source_dim_idx;
        }

        return std::make_shared<Variable>(std::move(info));
    }
} // namespace xdataset
