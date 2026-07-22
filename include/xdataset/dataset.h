#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>

#include <tsl/ordered_map.h>

#include "block.h"
#include "data_array.h"

namespace xdataset 
{
    // ========================================================================
    // Dataset
    // ========================================================================
    //
    // A Dataset models a named collection of Analyses.  Each Analysis may
    // contain results from multiple simulation types (e.g. SP, HB, Transient).
    //
    // Full hierarchical path:
    //
    //    dataset . analysis . result_type . data_array
    //
    // Example:  noise.SP1.SP.freq
    //
    //    noise  -- Dataset name
    //    SP1    -- Analysis name (e.g. a particular design variant)
    //    SP     -- Simulation result type (e.g. S-parameter sweep)
    //    freq   -- DataArray within the block  (independent / dependent var)
    //
    // The (analysis, result_type) pair uniquely identifies one Block.
    // The Block's own name is the full prefix:  "noise.SP1.SP".
    //
    // Shortcut: if `freq` is a unique DataArray name across the entire
    // Dataset, you can omit analysis and result type:
    //
    //    noise..freq   -- behaves like noise.*.*.freq
    //
    // ========================================================================

    class XDATASET_API Dataset
    {
    public:
        /// result_type (e.g. "SP", "HB") -> Block
        using ResultTypeMap = tsl::ordered_map<std::string, Block>;

        /// analysis_name -> ResultTypeMap
        using AnalysisMap = tsl::ordered_map<std::string, ResultTypeMap>;

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

        /// Add a Block identified by (analysis_name, result_type).
        ///
        /// The analysis is created if it does not yet exist.
        /// Returns a reference to the inserted Block.
        ///
        /// Example:  AddBlock("SP1", "SP", info)
        ///   -> creates Block "noise.SP1.SP" inside analysis "SP1".
        Block& AddBlock(const std::string& analysis_name,
                        const std::string& result_type,
                        const BlockCreateInfo& block_info);

        Block& AddBlock(const std::string& analysis_name,
                        const std::string& result_type,
                        BlockCreateInfo&& block_info);

        /// Remove an entire Analysis and every Block it contains.
        /// Returns 1 if removed, 0 if the analysis did not exist.
        std::size_t RemoveAnalysis(const std::string& analysis_name);

        /// Remove the Block for a given (analysis_name, result_type).
        /// If the owning Analysis becomes empty it is removed as well.
        /// Returns 1 if removed, 0 if not found.
        std::size_t RemoveBlock(const std::string& analysis_name,
                                const std::string& result_type);

        // --------------------------------------------------------------------
        // Query
        // --------------------------------------------------------------------

        bool HasAnalysis(const std::string& analysis_name) const;

        bool HasBlock(const std::string& analysis_name,
                      const std::string& result_type) const;

        /// Returns true when `data_array_name` appears exactly once across
        /// all blocks in the entire Dataset — the "..name" shortcut works.
        bool HasUniqueDataArray(const std::string& data_array_name) const;

        // --------------------------------------------------------------------
        // Access — explicit path
        // --------------------------------------------------------------------

        /// Full hierarchical access:  dataset.analysis.result_type.data_array
        ///
        /// Example:  GetDataArray("SP1", "SP", "freq")  ->  noise.SP1.SP.freq
        ///
        /// Non-const because a Block may lazily create its DataArray cache.
        const DataArray& GetDataArray(const std::string& analysis_name,
                                      const std::string& result_type,
                                      const std::string& data_array_name);

        /// Shortcut: look up a DataArray by its unique name alone.
        ///
        /// Throws std::invalid_argument if the name is not found or is not
        /// unique.  Equivalent to noise..freq when `freq` is unambiguous.
        const DataArray& GetDataArray(const std::string& data_array_name);

        /// All result-type blocks inside one Analysis (const).
        const ResultTypeMap& GetAnalysis(
            const std::string& analysis_name) const;

        /// Single Block by (analysis, result_type) (const).
        const Block& GetBlock(const std::string& analysis_name,
                              const std::string& result_type) const;

        /// Mutable access — allows lazy DataArray creation inside Block
        /// (via Block::GetOrCreateDataArray).
        ResultTypeMap& GetAnalysis(const std::string& analysis_name);
        Block&         GetBlock(const std::string& analysis_name,
                                const std::string& result_type);

        // --------------------------------------------------------------------
        // Enumeration
        // --------------------------------------------------------------------

        /// Ordered analysis names (insertion order).
        std::vector<std::string> GetAnalysisNames() const;

        /// Ordered DataArray names within a specific Block
        /// (independents first, then dependents, insertion order).
        std::vector<std::string> GetDataArrayNames(
            const std::string& analysis_name,
            const std::string& result_type) const;

        // --------------------------------------------------------------------
        // Capacity
        // --------------------------------------------------------------------

        std::size_t analysis_count() const { return analyses_.size(); }

    private:
        /// Walk every Block in every Analysis looking for `data_array_name`.
        ///
        /// Returns {analysis_name, result_type} of the unique match.
        /// Throws std::invalid_argument if zero or multiple matches exist.
        std::pair<std::string, std::string> FindUniqueDataArray(
            const std::string& data_array_name) const;

        std::string name_;
        AnalysisMap analyses_;
    };
}

#endif  // DATASET_H