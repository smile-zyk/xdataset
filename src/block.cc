#include "block.h"

#include <stdexcept>

namespace xdataset
{
    namespace
    {
        void ensure_unique_name(const tsl::ordered_map<std::string, VariableDescriptor>& descriptor_map,
                                const std::string& name)
        {
            if (name.empty())
            {
                throw std::invalid_argument("variable name must not be empty");
            }
            if (descriptor_map.find(name) != descriptor_map.end())
            {
                throw std::invalid_argument("duplicate variable name in block: " + name);
            }
        }

    } // namespace

    Block::Block(const BlockCreateInfo& info)
                : name_(info.name),
                    dependents_(),
                    independents_(),
                    variable_descriptor_map_(),
                    variable_cache_(),
                    table_data_cache_valid_(false),
                    table_data_cache_()
    {
        std::vector<DimensionSpec> independent_dims;
        independent_dims.reserve(info.independent_variables.size());

        for (const auto& independent : info.independent_variables)
        {
            ensure_unique_name(variable_descriptor_map_, independent.name);

            VariableDescriptorCreateInfo descriptor_info;
            descriptor_info.name = independent.name;
            descriptor_info.data = independent.data;
            descriptor_info.kind = VariableKind::kIndependent;
            descriptor_info.multi_dimension_spec.add_dimension(independent.dimension);

            variable_descriptor_map_.emplace(independent.name, VariableDescriptor(descriptor_info));
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
            ensure_unique_name(variable_descriptor_map_, dependent.name);

            VariableDescriptorCreateInfo descriptor_info;
            descriptor_info.name = dependent.name;
            descriptor_info.data = dependent.data;
            descriptor_info.kind = VariableKind::kDependent;
            descriptor_info.multi_dimension_spec = dependent_multi_dim;

            variable_descriptor_map_.emplace(dependent.name, VariableDescriptor(descriptor_info));
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

    const VariableDescriptor& Block::variable_descriptor(const std::string& name) const
    {
        auto it = variable_descriptor_map_.find(name);
        if (it == variable_descriptor_map_.end())
        {
            throw std::invalid_argument("variable descriptor not found: " + name);
        }
        return it->second;
    }

    std::shared_ptr<Variable> Block::CreateVariable(const VariableDescriptor& descriptor)
    {
        VariableCreateInfo info;
        info.name = descriptor.name();
        info.data = descriptor.data();
        info.kind = descriptor.kind();

        if (descriptor.kind() == VariableKind::kDependent)
        {
            info.multi_dimension_spec = descriptor.multi_dimension_spec();
            for (const std::string& indep_name : independents_)
            {
                info.indep_datas.emplace(indep_name, variable_descriptor(indep_name).data());
            }
            return std::make_shared<Variable>(std::move(info));
        }

        MultiDimensionSpec composed_multi_dim;
        bool found_current_independent = false;

        for (const std::string& indep_name : independents_)
        {
            const VariableDescriptor& indep_descriptor = variable_descriptor(indep_name);
            const MultiDimensionSpec& indep_dim = indep_descriptor.multi_dimension_spec();
            if (indep_dim.rank() != 1 || indep_dim.dims().size() != 1)
            {
                throw std::logic_error(
                    "independent VariableDescriptor must store exactly one dimension");
            }

            composed_multi_dim.add_dimension(indep_dim.dims()[0]);
            if (indep_name == descriptor.name())
            {
                found_current_independent = true;
                break;
            }

            info.indep_datas.emplace(indep_name, indep_descriptor.data());
        }

        if (!found_current_independent)
        {
            throw std::invalid_argument(
                "independent variable not found in block ordering: " + descriptor.name());
        }

        info.multi_dimension_spec = composed_multi_dim;
        return std::make_shared<Variable>(std::move(info));
    }

    std::shared_ptr<Variable> Block::GetOrCreateVariable(const std::string& variable_name)
    {
        auto cached_it = variable_cache_.find(variable_name);
        if (cached_it != variable_cache_.end())
        {
            return cached_it->second;
        }

        const VariableDescriptor& descriptor = variable_descriptor(variable_name);
        std::shared_ptr<Variable> variable = CreateVariable(descriptor);
        if (variable)
        {
            variable_cache_[variable_name] = variable;
        }
        return variable;
    }

    const TableData& Block::GetOrCreateTableData() const
    {
        if (table_data_cache_valid_)
        {
            return table_data_cache_;
        }

        TableData table;

        std::vector<DimensionSpec> dims;
        dims.reserve(independents_.size());
        for (const std::string& indep_name : independents_)
        {
            const VariableDescriptor& indep_descriptor = variable_descriptor(indep_name);
            const std::vector<std::string> headers =
                TableData::ExpandHeadersForSeries(indep_name, indep_descriptor.data());
            table.metadata.headers.insert(
                table.metadata.headers.end(), headers.begin(), headers.end());
            const MultiDimensionSpec& indep_dim = indep_descriptor.multi_dimension_spec();
            if (indep_dim.rank() != 1 || indep_dim.dims().size() != 1)
            {
                throw std::logic_error("independent VariableDescriptor must store exactly one dimension");
            }
            dims.push_back(indep_dim.dims()[0]);
        }
        for (const std::string& dep_name : dependents_)
        {
            const VariableDescriptor& dep_descriptor = variable_descriptor(dep_name);
            const std::vector<std::string> headers =
                TableData::ExpandHeadersForSeries(dep_name, dep_descriptor.data());
            table.metadata.headers.insert(
                table.metadata.headers.end(), headers.begin(), headers.end());
        }

        if (dims.empty())
        {
            table_data_cache_ = table;
            table_data_cache_valid_ = true;
            return table_data_cache_;
        }

        MultiDimensionSpec traversal_spec(dims);
        traversal_spec.for_each_leaf_row(
            [&](const MultiDimensionSpec::LeafRow& leaf_row)
            {
                const std::vector<std::size_t>& multi_index = leaf_row.multi_index;
                const std::vector<std::size_t>& dimension_row_indices =
                    leaf_row.dimension_row_indices;
                const std::size_t row_flat = leaf_row.row_flat;

                std::vector<std::string> row;
                row.reserve(table.metadata.headers.size());

                for (std::size_t dim = 0; dim < independents_.size(); ++dim)
                {
                    const VariableDescriptor& indep_descriptor = variable_descriptor(independents_[dim]);
                    const std::vector<std::string> values =
                        TableData::CellAtToColumns(indep_descriptor.data(), dimension_row_indices[dim]);
                    row.insert(row.end(), values.begin(), values.end());
                }

                for (const std::string& dep_name : dependents_)
                {
                    const VariableDescriptor& dep_descriptor = variable_descriptor(dep_name);
                    const std::vector<std::string> values =
                        TableData::CellAtToColumns(dep_descriptor.data(), row_flat);
                    row.insert(row.end(), values.begin(), values.end());
                }

                table.metadata.multi_indices.push_back(multi_index);
                table.rows.push_back(row);
            });

        table_data_cache_ = std::move(table);
        table_data_cache_valid_ = true;
        return table_data_cache_;
    }
} // namespace xdataset
