#ifndef DATASET_H
#define DATASET_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "block.h"
#include "data_array.h"

namespace xdataset
{

// =========================================================================
// Group — internal node in the Dataset tree
// =========================================================================
//
// When `block` is non-null, the Group is a leaf (holds a Block).
// When `block` is null, the Group is a pure container for sub-groups.
// Groups are created implicitly by Dataset::AddBlock().
// =========================================================================

struct XDATASET_API Group
{
    tsl::ordered_map<std::string, std::unique_ptr<Group>> subgroups;
    std::unique_ptr<Block> block;

    bool is_leaf()  const { return block != nullptr; }
    bool is_empty() const { return subgroups.empty() && block == nullptr; }
};

// ========================================================================
    // ========================================================================
    // Dataset
    // ========================================================================
    //
    // A Dataset is a tree-structured container.  Internal nodes are Groups
    // (no data, just name → child mappings).  Leaf nodes are Blocks which
    // hold the actual simulation data (independents + dependents).
    //
    // C++ API uses '/' as path separator; REL uses '.' for the tree and
    // a final '.' to separate block name from data_array name:
    //
    //    C++:  GetDataArray("simulation/SP1/SP", "Vout")
    //    REL:  noise.simulation.SP1.SP.Vout
    //
    //    noise            -- Dataset name
    //    simulation/SP1   -- nested Groups (created implicitly by AddBlock)
    //    SP               -- Block (leaf node, holds independents + dependents)
    //    Vout             -- DataArray within the Block
    //
    // Intermediate Groups are created on demand when AddBlock is called.
    //
    // Shortcut: if `Vout` is a unique DataArray name across the entire
    // Dataset, you can omit the block path:
    //
    //    C++:  GetDataArray("Vout")
    //    REL:  noise..Vout   -- matches *.*. ... .Vout (any block)
    //
    // ========================================================================

    class XDATASET_API Dataset
    {
    public:
        // --------------------------------------------------------------------
        // Construction
        // --------------------------------------------------------------------

        Dataset() = default;
        explicit Dataset(std::string name);

        /// Human-readable name, e.g. "noise".
        const std::string& name() const { return name_; }
        void               set_name(std::string name) { name_ = std::move(name); }

        // --------------------------------------------------------------------
        // Mutation
        // --------------------------------------------------------------------

        /// Add a Block at `path`.  Intermediate Groups are created implicitly.
        /// Block::name() is the path with `/` replaced by `.`.
        ///
        /// Example:  AddBlock("simulation/SP1/SP", info) → Block "simulation.SP1.SP"
        Block& AddBlock(const std::string& path,
                        const BlockCreateInfo& block_info);

        Block& AddBlock(const std::string& path,
                        BlockCreateInfo&& block_info);

        /// Remove a Block and return 1, or 0 if not found.  Empty parent
        /// Groups are NOT automatically cleaned up.
        std::size_t RemoveBlock(const std::string& path);

        /// Remove a Group and all its descendants.  Returns the number
        /// of Blocks removed.
        std::size_t RemoveGroup(const std::string& path);

        // --------------------------------------------------------------------
        // Query
        // --------------------------------------------------------------------

        /// True if a Block exists at `path`.
        bool HasBlock(const std::string& path) const;

        /// True if a Group (or leaf Block) exists at `path`.
        bool HasGroup(const std::string& path) const;

        /// True when `data_array_name` appears exactly once across all
        /// Blocks in the entire Dataset.
        bool HasUniqueDataArray(const std::string& data_array_name) const;

        // --------------------------------------------------------------------
        // Access
        // --------------------------------------------------------------------

        /// Full hierarchical access.
        ///
        /// Example:  GetDataArray("simulation/SP1/SP", "freq")
        const DataArray& GetDataArray(const std::string& block_path,
                                      const std::string& data_array_name);

        /// Global unique-name shortcut.
        /// Equivalent to `//data_array_name`.
        const DataArray& GetDataArray(const std::string& data_array_name);

        /// Const / mutable Block access by path.
        const Block& GetBlock(const std::string& path) const;
        Block&       GetBlock(const std::string& path);

        /// Ordered DataArray names within a specific Block
        /// (independents first, then dependents, insertion order).
        std::vector<std::string> GetDataArrayNames(
            const std::string& block_path) const;

        // --------------------------------------------------------------------
        // Enumeration
        // --------------------------------------------------------------------

        /// Direct child Block names under `group_path` (root = "").
        std::vector<std::string> GetBlockNames(
            const std::string& group_path = "") const;

        /// Direct child Group names under `group_path` (root = "").
        std::vector<std::string> GetGroupNames(
            const std::string& group_path = "") const;

        /// All Block paths (recursive), insertion order.
        std::vector<std::string> GetAllBlockPaths() const;

        // --------------------------------------------------------------------
        // Capacity
        // --------------------------------------------------------------------

        /// Total number of Blocks in the Dataset.
        std::size_t block_count() const;

    private:
        /// Navigate to the Group at `path` (const).  Returns nullptr if not found.
        const Group* navigate(const std::string& path) const;

        /// Navigate to the Group at `path` (mutable).  Returns nullptr if not found.
        Group* navigate(const std::string& path);

        /// Split "a/b/c" into ["a", "b", "c"].
        static std::vector<std::string> split_path(const std::string& path);

        /// Navigate, creating intermediate Groups as needed.
        Group& navigate_or_create(const std::string& path);

        /// Recursively collect Block names.
        void collect_block_paths(const Group& group,
                                 const std::string& prefix,
                                 std::vector<std::string>& paths) const;

        /// Recursively count Blocks.
        std::size_t collect_block_count(const Group& group) const;

        /// Walk the Group tree looking for `data_array_name`.
        /// Returns the block path of the unique match.
        std::string find_unique_data_array(
            const std::string& data_array_name) const;

        std::string name_;
        Group       root_;
    };
}

#endif  // DATASET_H