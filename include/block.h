#ifndef BLOCK_H
#define BLOCK_H

#include <memory>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

#include "data_series.h"
#include "dimension_spec.h"
#include "data_frame.h"
#include "data_array.h"

namespace xdataset
{
    /// True when `name` is a valid REL identifier: [A-Za-z_][A-Za-z0-9_]*.
    /// Dataset names and Block path segments must all satisfy this so that
    /// REL references (dataset.block.array) and the global source_path
    /// ("<dataset>/<block path>") remain unambiguous.
    inline bool IsValidIdentifier(const std::string& name)
    {
        if (name.empty()) return false;
        const char c0 = name[0];
        const bool start_ok =
            (c0 >= 'a' && c0 <= 'z') || (c0 >= 'A' && c0 <= 'Z') || c0 == '_';
        if (!start_ok) return false;
        for (std::size_t i = 1; i < name.size(); ++i)
        {
            const char c = name[i];
            const bool ok =
                (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_';
            if (!ok) return false;
        }
        return true;
    }

    /// True when `name` is a valid Block name: one or more identifiers
    /// joined by '/', e.g. "amplifier/DC/bias" -- the Block's full path
    /// within the Dataset tree.  A Block name equals its path; the
    /// source_path prefixes the Dataset name.  No empty segments and no
    /// leading/trailing '/' are allowed.
    inline bool IsValidBlockName(const std::string& name)
    {
        if (name.empty()) return false;
        if (name.front() == '/' || name.back() == '/') return false;
        std::size_t pos = 0;
        while (pos < name.size())
        {
            const std::size_t slash = name.find('/', pos);
            const std::string seg =
                name.substr(pos, slash == std::string::npos
                                     ? std::string::npos
                                     : slash - pos);
            if (!IsValidIdentifier(seg)) return false;
            if (slash == std::string::npos) break;
            pos = slash + 1;
        }
        return true;
    }

    struct IndependentSpec
    {
        std::string   name;
        DataSeries    data;
        DimensionSpec dimension;
    };

    struct DependentSpec
    {
        std::string name;
        DataSeries  data;
    };

    struct BlockCreateInfo
    {
        std::vector<IndependentSpec> independent_specs;
        std::vector<DependentSpec>   dependent_specs;
    };

    // ========================================================================
    // Block -- leaf node in the Dataset tree
    // ========================================================================
    //
    // A Block holds the independent variables (coordinate axes) and dependent
    // variables (measurements) for one simulation result.  It is always a
    // LEAF in the Dataset tree -- Blocks do not contain other Blocks.
    //
    // Block.name() returns the Block's full path within the Dataset, using
    // '/' separators, e.g. AddBlock("simulation/SP1/SP", info) ->
    // Block::name() == "simulation/SP1/SP".  The source_path() prefixes the
    // Dataset name ("<datasetName>/<block path>").
    //
    // The name is fixed at construction: it is assigned by Dataset::AddBlock
    // (from the path) and never changes afterwards.  External code cannot
    // rename a Block (set_name is private; only Dataset may assign it).
    //
    // ========================================================================
    class XDATASET_API Block
    {
        friend class Dataset;  // AddBlock assigns the immutable name.

    public:
        explicit Block(const BlockCreateInfo& info);
        explicit Block(BlockCreateInfo&& info);

        /// Construct with an explicit name.  The name is fixed at
        /// construction and can never be changed afterwards.
        Block(std::string name, const BlockCreateInfo& info);
        Block(std::string name, BlockCreateInfo&& info);

        /// Full path within the Dataset, e.g. "simulation/SP1/SP".
        const std::string& name() const;

        /// Globally-unique source path of this Block:
        /// "<datasetName>/<block path>" with '/' separators, e.g.
        /// "noise/simulation/SP1/SP".  Fixed at AddBlock time; used as the
        /// DataArray source_block_path for arrays created here.
        const std::string& source_path() const { return source_path_; }

        std::vector<std::string> dependents() const;

        std::vector<std::string> independents() const;

        const IndependentSpec& independent_spec(const std::string& name) const;

        const DependentSpec& dependent_spec(const std::string& name) const;

        const DataArray& GetOrCreateDataArray(const std::string& name) const;
        const DataFrame& GetOrCreateDataFrame() const;

    private:
        void set_name(std::string name);        // Dataset (friend) only, at AddBlock time.
        void set_source_path(std::string path); // Dataset (friend) only, at AddBlock time.

        DataArray CreateDataArray(const IndependentSpec& info) const;
        void ensure_unique_name(const std::string& name) const;

        std::string                                        name_;
        std::string                                        source_path_;
        tsl::ordered_map<std::string, IndependentSpec> independent_spec_map_;
        tsl::ordered_map<std::string, DependentSpec>   dependent_spec_map_;
        mutable tsl::ordered_map<std::string, std::unique_ptr<DataArray>> data_array_cache_;
        mutable std::unique_ptr<DataFrame>                    data_frame_cache_;
    };
}

#endif  // BLOCK_H