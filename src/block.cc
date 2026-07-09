#include "block.h"

#include <complex>
#include <sstream>
#include <stdexcept>

namespace xdataset
{
    namespace
    {
        void ensure_unique_name(const tsl::ordered_map<std::string, VariableSpec>& spec_map,
                                const std::string& name)
        {
            if (name.empty())
            {
                throw std::invalid_argument("variable name must not be empty");
            }
            if (spec_map.find(name) != spec_map.end())
            {
                throw std::invalid_argument("duplicate variable name in block: " + name);
            }
        }

        std::string FormatMultiIndex(const std::vector<std::size_t>& index)
        {
            std::ostringstream oss;
            oss << "[";
            for (std::size_t i = 0; i < index.size(); ++i)
            {
                if (i > 0)
                {
                    oss << ",";
                }
                oss << index[i];
            }
            oss << "]";
            return oss.str();
        }

        template <typename T>
        std::string NumberToString(const T& value)
        {
            std::ostringstream oss;
            oss << value;
            return oss.str();
        }

        std::string ComplexToString(const std::complex<double>& value)
        {
            std::ostringstream oss;
            oss << value.real() << "+" << value.imag() << "i";
            return oss.str();
        }

        std::string ScalarAtToString(const CellSeries& series, std::size_t row)
        {
            if (series.cell_kind() != CellKind::kScalar)
            {
                throw std::logic_error("block table view currently requires scalar CellSeries");
            }

            switch (series.dtype())
            {
                case DTypeTag::kReal:
                    return NumberToString(series.scalar_at<double>(row));
                case DTypeTag::kInteger:
                    return NumberToString(series.scalar_at<int>(row));
                case DTypeTag::kComplex:
                    return ComplexToString(series.scalar_at<std::complex<double>>(row));
                case DTypeTag::kString:
                    return series.scalar_at<std::string>(row);
            }
            throw std::logic_error("unsupported scalar dtype");
        }

    } // namespace

    Block::Block(const BlockCreateInfo& info)
        : name_(info.name), dependents_(), independents_(), variable_spec_map_(), variable_data_cache_()
    {
        std::vector<DimensionSpec> independent_dims;
        independent_dims.reserve(info.independent_variables.size());

        for (const auto& independent : info.independent_variables)
        {
            ensure_unique_name(variable_spec_map_, independent.name);

            VariableSpecCreateInfo spec_info;
            spec_info.name = independent.name;
            spec_info.data = independent.data;
            spec_info.kind = VariableKind::kIndependent;
            spec_info.multi_dimension_spec.add_dimension(independent.dimension);

            variable_spec_map_.emplace(independent.name, VariableSpec(spec_info));
            independents_.push_back(independent.name);
            independent_dims.push_back(independent.dimension);
        }

        if (!info.dependent_variables.empty() && independents_.empty())
        {
            throw std::invalid_argument(
                "dependent variables require at least one independent variable");
        }

        MultiDimensionSpec dependent_multi_dim;
        for (std::size_t i = 0; i < independent_dims.size(); ++i)
        {
            dependent_multi_dim.add_dimension(independent_dims[i]);
        }

        for (const auto& dependent : info.dependent_variables)
        {
            ensure_unique_name(variable_spec_map_, dependent.name);

            VariableSpecCreateInfo spec_info;
            spec_info.name = dependent.name;
            spec_info.data = dependent.data;
            spec_info.kind = VariableKind::kDependent;
            spec_info.multi_dimension_spec = dependent_multi_dim;

            variable_spec_map_.emplace(dependent.name, VariableSpec(spec_info));
            dependents_.push_back(dependent.name);
        }
    }

    Block::Block(BlockCreateInfo&& info)
        : Block(static_cast<const BlockCreateInfo&>(info))
    {
        name_ = std::move(info.name);
    }

    const std::string& Block::name() const
    {
        return name_;
    }

    const std::vector<std::string>& Block::dependents() const
    {
        return dependents_;
    }

    const std::vector<std::string>& Block::independents() const
    {
        return independents_;
    }

    const VariableSpec& Block::variable_spec(const std::string& name) const
    {
        auto it = variable_spec_map_.find(name);
        if (it == variable_spec_map_.end())
        {
            throw std::invalid_argument("variable spec not found: " + name);
        }
        return it->second;
    }

    std::shared_ptr<VariableData> Block::CreateVariableData(const VariableSpec& spec)
    {
        VariableDataCreateInfo info;
        info.name = spec.name();
        info.data = spec.data();
        info.kind = spec.kind();

        if (spec.kind() == VariableKind::kDependent)
        {
            info.multi_dimension_spec = spec.multi_dimension_spec();
            for (const std::string& indep_name : independents_)
            {
                info.indep_datas.emplace(indep_name, variable_spec(indep_name).data());
            }
            return std::make_shared<VariableData>(std::move(info));
        }

        MultiDimensionSpec composed_multi_dim;
        bool found_current_independent = false;

        for (const std::string& indep_name : independents_)
        {
            const VariableSpec& indep_spec = variable_spec(indep_name);
            const MultiDimensionSpec& indep_dim = indep_spec.multi_dimension_spec();
            if (indep_dim.rank() != 1 || indep_dim.dims().size() != 1)
            {
                throw std::logic_error(
                    "independent VariableSpec must store exactly one dimension");
            }

            composed_multi_dim.add_dimension(indep_dim.dims()[0]);
            if (indep_name == spec.name())
            {
                found_current_independent = true;
                break;
            }

            info.indep_datas.emplace(indep_name, indep_spec.data());
        }

        if (!found_current_independent)
        {
            throw std::invalid_argument(
                "independent variable not found in block ordering: " + spec.name());
        }

        info.multi_dimension_spec = composed_multi_dim;
        return std::make_shared<VariableData>(std::move(info));
    }

    std::shared_ptr<VariableData> Block::GetOrCreateVariableData(const std::string& variable_name)
    {
        auto cached_it = variable_data_cache_.find(variable_name);
        if (cached_it != variable_data_cache_.end())
        {
            return cached_it->second;
        }

        const VariableSpec& spec = variable_spec(variable_name);
        std::shared_ptr<VariableData> var_data = CreateVariableData(spec);
        if (var_data)
        {
            variable_data_cache_[variable_name] = var_data;
        }
        return var_data;
    }

    TableData Block::ToTableData()
    {
        TableData table;

        std::vector<DimensionSpec> dims;
        dims.reserve(independents_.size());
        for (const std::string& indep_name : independents_)
        {
            table.metadata.headers.push_back(indep_name);
            const VariableSpec& indep_spec = variable_spec(indep_name);
            const MultiDimensionSpec& indep_dim = indep_spec.multi_dimension_spec();
            if (indep_dim.rank() != 1 || indep_dim.dims().size() != 1)
            {
                throw std::logic_error("independent VariableSpec must store exactly one dimension");
            }
            dims.push_back(indep_dim.dims()[0]);
        }
        for (const std::string& dep_name : dependents_)
        {
            table.metadata.headers.push_back(dep_name);
        }

        if (dims.empty())
        {
            return table;
        }

        MultiDimensionSpec traversal_spec(dims);
        traversal_spec.for_each_leaf_row(
            [&](const std::vector<std::size_t>& multi_index,
                const std::vector<std::size_t>& dimension_row_indices,
                std::size_t row_flat)
            {
                std::vector<std::string> row;
                row.reserve(table.metadata.headers.size());

                for (std::size_t dim = 0; dim < independents_.size(); ++dim)
                {
                    const VariableSpec& indep_spec = variable_spec(independents_[dim]);
                    row.push_back(ScalarAtToString(indep_spec.data(), dimension_row_indices[dim]));
                }

                for (const std::string& dep_name : dependents_)
                {
                    const VariableSpec& dep_spec = variable_spec(dep_name);
                    row.push_back(ScalarAtToString(dep_spec.data(), row_flat));
                }

                table.metadata.multi_indices.push_back(multi_index);
                table.rows.push_back(row);
            });

        return table;
    }
} // namespace xdataset
