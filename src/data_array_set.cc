#include "data_array_set.h"

#include <stdexcept>

#include "xdataset_predefine.h"

namespace xdataset
{
    DataArraySet::DataArraySet(std::string name)
        : name_(std::move(name))
    {
    }

    DataArray& DataArraySet::Add(const std::string& name, DataArray array)
    {
        if (!IsValidIdentifier(name))
            throw std::invalid_argument(
                "DataArraySet member name must be a valid identifier: " + name);

        // Replace in place (preserving the original insertion position) or
        // insert a new entry.
        auto it = arrays_.find(name);
        if (it != arrays_.end())
        {
            *it->second = std::move(array);
            return *it->second;
        }

        auto owned = std::unique_ptr<DataArray>(new DataArray(std::move(array)));
        DataArray& ref = *owned;
        arrays_.emplace(name, std::move(owned));
        return ref;
    }

    std::size_t DataArraySet::Remove(const std::string& name)
    {
        return arrays_.erase(name);
    }

    bool DataArraySet::Has(const std::string& name) const
    {
        return arrays_.count(name) != 0;
    }

    const DataArray& DataArraySet::Get(const std::string& name) const
    {
        auto it = arrays_.find(name);
        if (it == arrays_.end())
            throw std::out_of_range("DataArray not found in set: " + name);
        return *it->second;
    }

    DataArray& DataArraySet::Get(const std::string& name)
    {
        auto it = arrays_.find(name);
        if (it == arrays_.end())
            throw std::out_of_range("DataArray not found in set: " + name);
        return *it->second;
    }

    std::vector<std::string> DataArraySet::names() const
    {
        std::vector<std::string> out;
        out.reserve(arrays_.size());
        for (const auto& kv : arrays_)
            out.push_back(kv.first);
        return out;
    }

} // namespace xdataset
