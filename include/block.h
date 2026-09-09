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
    // '.' separators, e.g. AddBlock("simulation.SP1.SP", info) ->
    // Block::name() == "simulation.SP1.SP".  The source_path() prefixes the
    // Dataset name ("<datasetName>.<block path>").
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

        /// Reconstruct a set of independent coordinate specs from tabular
        /// coordinate rows.  This is the counterpart of the DataFrame tabular
        /// view: it recovers each independent axis (name + coordinate values +
        /// DimensionSpec) from the row-major coordinate cells.
        ///
        /// @param rows          One entry per row (row-major traversal order).
        ///                      Each inner vector holds the independent
        ///                      coordinate values for that row, in dimension
        ///                      order (outermost dimension first).  All rows
        ///                      must share the same size == rank.
        /// @param column_names  Names of the independent axes, one per
        ///                      dimension.  If empty, defaults are generated
        ///                      ("dim0", "dim1", ...).  Otherwise its size must
        ///                      equal rank.
        ///
        /// The reconstruction is a top-down run-length decomposition over the
        /// coordinate columns:
        ///   - dim 0 is always Regular (a single root group);
        ///   - a dimension is Regular iff, for every parent group, the number
        ///     of child segments is identical AND the child coordinate values
        ///     are shared across all parents (a Cartesian product);
        ///   - otherwise the dimension is Ragged, with sizes = the child
        ///     counts of each parent group.
        ///
        /// Each returned IndependentSpec holds the axis in COMPACT form: its
        /// data length equals the corresponding DimensionSpec::element_count()
        /// (regular_size for Regular, prefix_sum().back() for Ragged), so the
        /// spec can be passed directly into BlockCreateInfo.
        ///
        /// Coordinates are compared by their string representation, so each
        /// Measurement only needs a unique textual form.  Scalars of any dtype
        /// are supported.
        ///
        /// @pre  rank >= 1 and rows is non-empty.
        /// @pre  Within each parent group, distinct child nodes map to
        ///       distinct coordinate values (an injective "child -> value"
        ///       mapping).  Violating this under-counts child segments.
        static std::vector<IndependentSpec> FromCoordinates(
            const std::vector<std::vector<Measurement>>& rows,
            std::vector<std::string> column_names = {});

        /// Full path within the Dataset, e.g. "simulation.SP1.SP".
        const std::string& name() const;

        /// Globally-unique source path of this Block:
        /// "<datasetName>.<block path>" with '.' separators, e.g.
        /// "noise.simulation.SP1.SP".  Fixed at AddBlock time; used as the
        /// DataArray source_block_path for arrays created here.
        const std::string& source_path() const { return source_path_; }

        /// Name of the Dataset that owns this Block, e.g. "noise".
        /// Fixed at AddBlock time.  Empty for Blocks constructed in-memory
        /// but not yet added to a Dataset.
        const std::string& dataset_name() const { return dataset_name_; }

        std::vector<std::string> dependents() const;

        std::vector<std::string> independents() const;

        const IndependentSpec& independent_spec(const std::string& name) const;

        const DependentSpec& dependent_spec(const std::string& name) const;

        const DataArray& GetOrCreateDataArray(const std::string& name) const;
        const DataFrame& GetOrCreateDataFrame() const;

    private:
        void set_name(std::string name);        // Dataset (friend) only, at AddBlock time.
        void set_source_path(std::string path); // Dataset (friend) only, at AddBlock time.
        void set_dataset_name(std::string name);// Dataset (friend) only, at AddBlock time.

        DataArray CreateDataArray(const IndependentSpec& info) const;
        void ensure_unique_name(const std::string& name) const;

        std::string                                        name_;
        std::string                                        source_path_;
        std::string                                        dataset_name_;
        tsl::ordered_map<std::string, IndependentSpec> independent_spec_map_;
        tsl::ordered_map<std::string, DependentSpec>   dependent_spec_map_;
        mutable tsl::ordered_map<std::string, std::unique_ptr<DataArray>> data_array_cache_;
        mutable std::unique_ptr<DataFrame>                    data_frame_cache_;
    };
}

#endif  // BLOCK_H