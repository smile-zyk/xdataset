#include "block.h"

#include <stdexcept>

namespace xdataset
{
    void Block::ensure_unique_name(const std::string& name) const
    {
        if (name.empty())
            throw std::invalid_argument("variable name must not be empty");
        if (independent_variable_map_.find(name) != independent_variable_map_.end())
            throw std::invalid_argument("duplicate variable name in block: " + name);
        if (dependent_variable_map_.find(name) != dependent_variable_map_.end())
            throw std::invalid_argument("duplicate variable name in block: " + name);
    }

    Block::Block(const BlockCreateInfo& info)
        : name_(info.name)
    {
        MultiDimensionSpec dependent_multi_dim;   // builds up from all independents in order

        for (const auto& iv : info.independent_variables)
        {
            ensure_unique_name(iv.name);

            independent_variable_map_.emplace(iv.name, iv);

            // Add in order — add_dimension validates ragged against prior dims.
            dependent_multi_dim.add_dimension(iv.dimension);

            // Validate: independent data size must match dimension element count.
            const std::size_t dim_elems = iv.dimension.is_regular()
                ? iv.dimension.regular_size()
                : iv.dimension.prefix_sum().back();
            if (iv.data.size() != static_cast<Index>(dim_elems))
                throw std::invalid_argument(
                    "independent variable '" + iv.name + "': data size " +
                    std::to_string(iv.data.size()) + " does not match dimension element count " +
                    std::to_string(dim_elems));
        }

        if (!info.dependent_variables.empty() && independent_variable_map_.empty())
            throw std::invalid_argument("dependent variables require at least one independent variable");

        for (const auto& dv : info.dependent_variables)
        {
            ensure_unique_name(dv.name);

            // Validate: dependent data size must match the product of independent dims.
            const std::size_t expected = dependent_multi_dim.compute_cell_count();
            if (dv.data.size() != static_cast<Index>(expected))
                throw std::invalid_argument(
                    "dependent variable '" + dv.name + "': data size " +
                    std::to_string(dv.data.size()) + " does not match derived cell count " +
                    std::to_string(expected));

            dependent_variable_map_.emplace(dv.name, dv);
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

    std::vector<std::string> Block::dependents() const
    {
        std::vector<std::string> names;
        names.reserve(dependent_variable_map_.size());
        for (const auto& kv : dependent_variable_map_)
            names.push_back(kv.first);
        return names;
    }

    std::vector<std::string> Block::independents() const
    {
        std::vector<std::string> names;
        names.reserve(independent_variable_map_.size());
        for (const auto& kv : independent_variable_map_)
            names.push_back(kv.first);
        return names;
    }

    const IndependentVariableInfo& Block::independent_variable(const std::string& name) const
    {
        auto it = independent_variable_map_.find(name);
        if (it == independent_variable_map_.end())
            throw std::invalid_argument("independent variable not found: " + name);
        return it->second;
    }

    const DependentVariableInfo& Block::dependent_variable(const std::string& name) const
    {
        auto it = dependent_variable_map_.find(name);
        if (it == dependent_variable_map_.end())
            throw std::invalid_argument("dependent variable not found: " + name);
        return it->second;
    }

    std::shared_ptr<Variable> Block::CreateVariable(const IndependentVariableInfo& info) const
    {
        VariableCreateInfo vinfo;
        vinfo.name = info.name;
        vinfo.kind = VariableKind::kIndependent;

        MultiDimensionSpec composed_multi_dim;
        std::vector<const CellSeries*> prior_indeps;
        bool found = false;

        for (const auto& kv : independent_variable_map_)
        {
            const IndependentVariableInfo& iv = kv.second;
            composed_multi_dim.add_dimension(iv.dimension);

            if (kv.first == info.name)
            {
                found = true;
                break;
            }
            prior_indeps.push_back(&iv.data);
        }

        if (!found)
            throw std::invalid_argument("independent variable not found in block ordering: " + info.name);

        // Expand own data to the full cartesian product.
        CellSeries expanded = CellSeries(info.data.cell_kind(), info.data.dtype(), info.data.cell_shape());
        composed_multi_dim.for_each_leaf_row(
            [&](const MultiDimensionSpec::LeafRow& lr)
            {
                expanded.append_from(info.data, lr.dimension_row_indices.back());
            });
        vinfo.data = std::move(expanded);

        // Prior independents stay raw.
        for (std::size_t p = 0; p < prior_indeps.size(); ++p)
        {
            auto it = independent_variable_map_.begin();
            std::advance(it, static_cast<std::ptrdiff_t>(p));
            vinfo.indep_datas.emplace(it->first, *prior_indeps[p]);
        }

        // Self: raw.
        vinfo.indep_datas.emplace(info.name, info.data);
        vinfo.multi_dimension_spec = composed_multi_dim;
        return std::make_shared<Variable>(std::move(vinfo));
    }

    std::shared_ptr<Variable> Block::GetOrCreateVariable(const std::string& variable_name)
    {
        auto cached_it = variable_cache_.find(variable_name);
        if (cached_it != variable_cache_.end())
            return cached_it->second;

        // Try independent first.
        auto it = independent_variable_map_.find(variable_name);
        if (it != independent_variable_map_.end())
        {
            auto var = CreateVariable(it->second);
            if (var)
                variable_cache_[variable_name] = var;
            return var;
        }

        // Dependent.
        auto dit = dependent_variable_map_.find(variable_name);
        if (dit != dependent_variable_map_.end())
        {
            VariableCreateInfo vinfo;
            vinfo.name = dit->second.name;
            vinfo.data = dit->second.data;
            vinfo.kind = VariableKind::kDependent;

            MultiDimensionSpec multi_dim;
            for (const auto& kv : independent_variable_map_)
            {
                const DimensionSpec& dim = kv.second.dimension;
                multi_dim.add_dimension(dim);
                vinfo.indep_datas.emplace(kv.first, kv.second.data);
            }
            vinfo.multi_dimension_spec = multi_dim;

            auto var = std::make_shared<Variable>(std::move(vinfo));
            variable_cache_[variable_name] = var;
            return var;
        }

        throw std::invalid_argument("variable not found: " + variable_name);
    }

    const GridModel& Block::grid_model() const
    {
        if (!grid_model_cache_)
            grid_model_cache_.reset(new BlockGridModel(*this));
        return *grid_model_cache_;
    }
} // namespace xdataset
