#include "block.h"

#include <stdexcept>

namespace xdataset
{
    void Block::ensure_unique_name(const std::string& name) const
    {
        if (name.empty())
            throw std::invalid_argument("DataArray name must not be empty");
        if (independent_spec_map_.find(name) != independent_spec_map_.end())
            throw std::invalid_argument("duplicate DataArray name in block: " + name);
        if (dependent_spec_map_.find(name) != dependent_spec_map_.end())
            throw std::invalid_argument("duplicate DataArray name in block: " + name);
    }

    Block::Block(const BlockCreateInfo& info)
        : name_(info.name)
    {
        MultiDimensionSpec dependent_multi_dim;   // builds up from all independents in order

        for (const auto& iv : info.independent_specs)
        {
            ensure_unique_name(iv.name);

            independent_spec_map_.emplace(iv.name, iv);

            // Add in order �� add_dimension validates ragged against prior dims.
            dependent_multi_dim.add_dimension(iv.dimension);

            // Validate: independent data size must match dimension element count.
            const std::size_t dim_elems = iv.dimension.is_regular()
                ? iv.dimension.regular_size()
                : iv.dimension.prefix_sum().back();
            if (iv.data.size() != static_cast<Index>(dim_elems))
                throw std::invalid_argument(
                    "independent DataArray '" + iv.name + "': data size " +
                    std::to_string(iv.data.size()) + " does not match dimension element count " +
                    std::to_string(dim_elems));
        }

        if (!info.dependent_specs.empty() && independent_spec_map_.empty())
            throw std::invalid_argument("dependent variables require at least one independent DataArray");

        for (const auto& dv : info.dependent_specs)
        {
            ensure_unique_name(dv.name);

            // Validate: dependent data size must match the product of independent dims.
            const std::size_t expected = dependent_multi_dim.compute_cell_count();
            if (dv.data.size() != static_cast<Index>(expected))
                throw std::invalid_argument(
                    "dependent DataArray '" + dv.name + "': data size " +
                    std::to_string(dv.data.size()) + " does not match derived cell count " +
                    std::to_string(expected));

            dependent_spec_map_.emplace(dv.name, dv);
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
        names.reserve(dependent_spec_map_.size());
        for (const auto& kv : dependent_spec_map_)
            names.push_back(kv.first);
        return names;
    }

    std::vector<std::string> Block::independents() const
    {
        std::vector<std::string> names;
        names.reserve(independent_spec_map_.size());
        for (const auto& kv : independent_spec_map_)
            names.push_back(kv.first);
        return names;
    }

    const IndependentSpec& Block::independent_spec(const std::string& name) const
    {
        auto it = independent_spec_map_.find(name);
        if (it == independent_spec_map_.end())
            throw std::invalid_argument("independent DataArray not found: " + name);
        return it->second;
    }

    const DependentSpec& Block::dependent_spec(const std::string& name) const
    {
        auto it = dependent_spec_map_.find(name);
        if (it == dependent_spec_map_.end())
            throw std::invalid_argument("dependent DataArray not found: " + name);
        return it->second;
    }

    DataArray Block::CreateDataArray(const IndependentSpec& info) const
    {
        DataArrayCreateInfo vinfo;
        vinfo.name = info.name;
        vinfo.kind = DataArrayKind::kIndependent;

        MultiDimensionSpec composed_multi_dim;
        std::vector<const DataSeries*> prior_indeps;
        bool found = false;

        for (const auto& kv : independent_spec_map_)
        {
            const IndependentSpec& iv = kv.second;
            composed_multi_dim.add_dimension(iv.dimension);

            if (kv.first == info.name)
            {
                found = true;
                break;
            }
            prior_indeps.push_back(&iv.data);
        }

        if (!found)
            throw std::invalid_argument("independent DataArray not found in block ordering: " + info.name);

        // Expand own data to the full cartesian product.
        DataSeries expanded = DataSeries(info.data.data_kind(), info.data.dtype(), info.data.data_shape());
        composed_multi_dim.for_each_leaf_row(
            [&](const MultiDimensionSpec::LeafRow& lr)
            {
                expanded.append_from(info.data, lr.dimension_row_indices.back());
            });
        vinfo.data = std::move(expanded);

        // Prior independents stay raw.
        for (std::size_t p = 0; p < prior_indeps.size(); ++p)
        {
            auto it = independent_spec_map_.begin();
            std::advance(it, static_cast<std::ptrdiff_t>(p));
            vinfo.indep_datas.emplace(it->first, *prior_indeps[p]);
        }

        // Self: raw.
        vinfo.indep_datas.emplace(info.name, info.data);
        vinfo.multi_dimension_spec = composed_multi_dim;
        return DataArray(std::move(vinfo));
    }

    const DataArray& Block::GetOrCreateDataArray(const std::string& name)
    {
        auto cached_it = data_array_cache_.find(name);
        if (cached_it != data_array_cache_.end())
            return *cached_it->second;

        // Try independent first.
        auto it = independent_spec_map_.find(name);
        if (it != independent_spec_map_.end())
        {
            std::unique_ptr<DataArray> var(new DataArray(CreateDataArray(it->second)));
            auto& ref = *var;
            data_array_cache_[name] = std::move(var);
            return ref;
        }

        // Dependent.
        auto dit = dependent_spec_map_.find(name);
        if (dit != dependent_spec_map_.end())
        {
            DataArrayCreateInfo vinfo;
            vinfo.name = dit->second.name;
            vinfo.data = dit->second.data;
            vinfo.kind = DataArrayKind::kDependent;

            MultiDimensionSpec multi_dim;
            for (const auto& kv : independent_spec_map_)
            {
                const DimensionSpec& dim = kv.second.dimension;
                multi_dim.add_dimension(dim);
                vinfo.indep_datas.emplace(kv.first, kv.second.data);
            }
            vinfo.multi_dimension_spec = multi_dim;

            std::unique_ptr<DataArray> var(new DataArray(std::move(vinfo)));
            auto& ref = *var;
            data_array_cache_[name] = std::move(var);
            return ref;
        }

        throw std::invalid_argument("DataArray not found: " + name);
    }

    const DataFrame& Block::GetOrCreateDataFrame() const
    {
        if (!data_frame_cache_)
            data_frame_cache_.reset(new BlockDataFrame(*this));
        return *data_frame_cache_;
    }
} // namespace xdataset
