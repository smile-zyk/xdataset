#ifndef XDATASET_DATA_ARRAY_SET_H
#define XDATASET_DATA_ARRAY_SET_H

#include <memory>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

#include "data_array.h"

namespace xdataset
{
    // ========================================================================
    // DataArraySet -- a leaf node holding named, unrelated DataArrays
    // ========================================================================
    //
    // A DataArraySet is a sibling of Block: both are leaf nodes in the
    // Dataset tree.  Whereas a Block models one simulation result
    // (independent coordinate axes + dependent measurements), a DataArraySet
    // is an insertion-ordered collection of arbitrary DataArrays that need
    // not share coordinate axes, shape, kind, or units.
    //
    // It participates in REL dotted access exactly like a Block:
    //
    //     REL:  noise.my_set.foo      -- DataArray "foo" inside "my_set"
    //     C++:  ds.GetDataArray("my_set", "foo")
    //
    // and can be persisted / restored alongside Blocks by DatasetIO (HDF5).
    //
    // ========================================================================
    class XDATASET_API DataArraySet
    {
    public:
        DataArraySet() = default;
        explicit DataArraySet(std::string name);

        /// Full path within the Dataset (assigned by Dataset::AddDataArraySet).
        const std::string& name() const { return name_; }

        /// "<datasetName>.<path>" with '.' separators (assigned by Dataset).
        const std::string& source_path() const { return source_path_; }

        /// Insert or replace a DataArray under `name`.  The name is validated
        /// as an identifier so REL references stay unambiguous.  Returns a
        /// reference to the stored array.
        DataArray& Add(const std::string& name, DataArray array);

        /// Remove a DataArray by name; returns 1 if removed, 0 if absent.
        std::size_t Remove(const std::string& name);

        /// True when a DataArray named `name` exists.
        bool Has(const std::string& name) const;

        /// Const / mutable access to a DataArray by name.
        /// Throws std::out_of_range when absent.
        const DataArray& Get(const std::string& name) const;
        DataArray& Get(const std::string& name);

        /// Ordered names of the contained DataArrays.
        std::vector<std::string> names() const;

        /// Number of DataArrays in the set.
        std::size_t size() const { return arrays_.size(); }

    private:
        friend class Dataset;  // AddDataArraySet assigns the immutable name.

        void set_name(std::string name) { name_ = std::move(name); }
        void set_source_path(std::string path) { source_path_ = std::move(path); }

        std::string name_;
        std::string source_path_;
        tsl::ordered_map<std::string, std::unique_ptr<DataArray>> arrays_;
    };

} // namespace xdataset

#endif // XDATASET_DATA_ARRAY_SET_H
