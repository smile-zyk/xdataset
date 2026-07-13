#include "variable.h"

#include <algorithm>
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

    std::shared_ptr<Variable> Variable::at(
        const std::vector<MultiIndexSelector>& selectors) const
    {
        return select(selectors);
    }

    std::shared_ptr<Variable> Variable::select(
        const std::vector<MultiIndexSelector>& selectors) const
    {
        if (selectors.size() != multi_dimension_spec_.rank())
        {
            throw std::invalid_argument("number of selectors must match variable rank");
        }

        const std::size_t indep_count = indep_datas_.size();

        std::vector<MultiIndexSelector> projection_selectors = selectors;
        if (kind_ == VariableKind::kIndependent)
        {
            const std::size_t self_dim = indep_count;
            const MultiIndexSelector& self_selector = projection_selectors[self_dim];
            if (self_selector.is_equal())
            {
                // Keep the self dimension in projected spec while selecting a single value.
                projection_selectors[self_dim] =
                    MultiIndexSelector::In(std::vector<Index>{self_selector.equal_value()});
            }
        }

        const MultiDimensionSpec::SelectionProjection projection =
            multi_dimension_spec_.project_selection(projection_selectors);

        VariableCreateInfo info;
        info.name = name_;
        info.kind = kind_;
        info.multi_dimension_spec = MultiDimensionSpec(projection.projected_dims);

        std::vector<const std::pair<std::string, CellSeries>*> indep_items;
        indep_items.reserve(indep_count);
        for (const auto& item : indep_datas_)
        {
            indep_items.push_back(&item);
        }

        std::vector<std::size_t> retained_indep_dims;
        retained_indep_dims.reserve(projection.retained_dims.size());
        for (std::size_t i = 0; i < projection.retained_dims.size(); ++i)
        {
            const std::size_t dim = projection.retained_dims[i];
            if (dim < indep_count)
            {
                retained_indep_dims.push_back(dim);
            }
        }

        std::vector<CellSeries> selected_indep_series;
        selected_indep_series.reserve(retained_indep_dims.size());
        std::vector<std::vector<bool> > selected_indep_seen;
        selected_indep_seen.reserve(retained_indep_dims.size());

        for (std::size_t i = 0; i < retained_indep_dims.size(); ++i)
        {
            const std::size_t dim = retained_indep_dims[i];
            const CellSeries& src = indep_items[dim]->second;
            selected_indep_series.push_back(
                CellSeries(src.cell_kind(), src.dtype(), src.cell_shape()));
            selected_indep_seen.push_back(std::vector<bool>(src.size(), false));
        }

        CellSeries selected_data(data_.cell_kind(), data_.dtype(), data_.cell_shape());
        std::vector<bool> selected_self_seen(data_.size(), false);

        for (std::size_t hit = 0; hit < projection.selected_rows.size(); ++hit)
        {
            const MultiDimensionSpec::LeafRow& selected_row = projection.selected_rows[hit];
            const std::vector<std::size_t>& multi_index = selected_row.multi_index;
            const std::vector<std::size_t>& dimension_row_indices =
                selected_row.dimension_row_indices;
            const std::size_t row_flat = selected_row.row_flat;

            for (std::size_t i = 0; i < retained_indep_dims.size(); ++i)
            {
                const std::size_t dim = retained_indep_dims[i];
                const CellSeries& src = indep_items[dim]->second;
                const std::size_t src_row = multi_index[dim];
                if (src_row >= src.size())
                {
                    throw std::out_of_range("selected independent row index out of bounds");
                }
                if (!selected_indep_seen[i][src_row])
                {
                    selected_indep_series[i].append_from(src, src_row);
                    selected_indep_seen[i][src_row] = true;
                }
            }

            if (kind_ == VariableKind::kIndependent)
            {
                const std::size_t self_dim = indep_count;
                const std::size_t self_row = dimension_row_indices[self_dim];
                if (self_row >= data_.size())
                {
                    throw std::out_of_range("selected independent self row index out of bounds");
                }
                if (!selected_self_seen[self_row])
                {
                    selected_data.append_from(data_, self_row);
                    selected_self_seen[self_row] = true;
                }
                continue;
            }

            if (row_flat >= data_.size())
            {
                throw std::out_of_range("selected dependent row index out of bounds");
            }
            selected_data.append_from(data_, row_flat);
        }

        for (std::size_t i = 0; i < retained_indep_dims.size(); ++i)
        {
            info.indep_datas.emplace(
                indep_items[retained_indep_dims[i]]->first,
                std::move(selected_indep_series[i]));
        }

        info.data = std::move(selected_data);

        return std::make_shared<Variable>(std::move(info));
    }
} // namespace xdataset
