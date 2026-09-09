#include "block.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace xdataset
{
    namespace
    {
        // A contiguous run of equal coordinate values within one column's
        // [start, end) row interval.  Ends where the value changes.
        struct CoordSegment
        {
            std::size_t begin;   // index into the column where this segment starts
            Measurement value;   // the (equal) coordinate value
            std::size_t length;  // number of consecutive rows
        };

        // Run-length encode col[start..end) on a single coordinate column.
        // Coords are compared with Measurement::operator==, so any scalar dtype
        // works (element-wise value comparison, unit-independent).
        std::vector<CoordSegment> run_length_encode(
            const std::vector<Measurement>& col,
            std::size_t start,
            std::size_t end)
        {
            std::vector<CoordSegment> segs;
            if (start >= end)
                return segs;

            segs.push_back(CoordSegment{start, col[start], 1});
            for (std::size_t i = start + 1; i < end; ++i)
            {
                if (col[i] == segs.back().value)
                {
                    ++segs.back().length;
                }
                else
                {
                    segs.push_back(CoordSegment{i, col[i], 1});
                }
            }
            return segs;
        }

        // File-local validator for Block names (paths).  Kept out of the
        // public header: not part of the API surface.
        bool IsValidBlockName(const std::string& name)
        {
            constexpr char sep = '.';
            if (name.empty()) return false;
            if (name.front() == sep || name.back() == sep) return false;
            std::size_t pos = 0;
            while (pos < name.size())
            {
                const std::size_t sep_pos = name.find(sep, pos);
                const std::string seg =
                    name.substr(pos, sep_pos == std::string::npos
                                         ? std::string::npos
                                         : sep_pos - pos);
                if (!IsValidIdentifier(seg)) return false;
                if (sep_pos == std::string::npos) break;
                pos = sep_pos + 1;
            }
            return true;
        }
    } // namespace

    std::vector<IndependentSpec> Block::FromCoordinates(
        const std::vector<std::vector<Measurement>>& rows,
        std::vector<std::string> column_names)
    {
        if (rows.empty())
            throw std::invalid_argument("Block::FromCoordinates: rows must not be empty");

        const std::size_t rank = rows[0].size();
        if (rank == 0)
            throw std::invalid_argument("Block::FromCoordinates: rows must have rank >= 1");
        for (const auto& r : rows)
        {
            if (r.size() != rank)
                throw std::invalid_argument(
                    "Block::FromCoordinates: all rows must have the same size (rank)");
        }

        // Column names are defaulted when omitted.
        if (column_names.empty())
        {
            column_names.reserve(rank);
            for (std::size_t d = 0; d < rank; ++d)
                column_names.push_back("dim" + std::to_string(d));
        }
        else if (column_names.size() != rank)
        {
            throw std::invalid_argument(
                "Block::FromCoordinates: column_names size must equal rank");
        }

        const std::size_t total_rows = rows.size();

        // Dimension d extracts its coordinate column from rows[i][d].
        // Transposed view: col[d][i] == rows[i][d].
        std::vector<std::vector<Measurement>> cols(rank);
        for (std::size_t d = 0; d < rank; ++d)
        {
            cols[d].reserve(total_rows);
            for (std::size_t i = 0; i < total_rows; ++i)
                cols[d].push_back(rows[i][d]);
        }

        // Top-down run-length decomposition.  Each level tracks the row
        // intervals of the next dimension's parent groups.
        struct ParentRange { std::size_t start; std::size_t end; };
        std::vector<ParentRange> parents;
        parents.push_back(ParentRange{0, total_rows});

        // Per-dimension compact coordinate columns and the recovered spec.
        std::vector<DataSeries>   axis_data;   // compact per-axis values
        std::vector<DimensionSpec> dims;

        axis_data.reserve(rank);
        dims.reserve(rank);

        for (std::size_t d = 0; d < rank; ++d)
        {
            const std::vector<Measurement>& col = cols[d];

            std::vector<std::size_t> child_counts;   // child segments per parent
            std::vector<std::vector<Measurement>> compact_by_parent;
            std::vector<ParentRange> next_parents;

            child_counts.reserve(parents.size());
            compact_by_parent.reserve(parents.size());
            next_parents.reserve(parents.size());

            for (const ParentRange& p : parents)
            {
                const std::vector<CoordSegment> segs =
                    run_length_encode(col, p.start, p.end);

                child_counts.push_back(segs.size());
                compact_by_parent.emplace_back(std::vector<Measurement>());
                compact_by_parent.back().reserve(segs.size());
                for (const CoordSegment& s : segs)
                    compact_by_parent.back().push_back(s.value);   // segment head value

                // Build next-level parent ranges from segment boundaries.
                std::size_t cur = p.start;
                for (const CoordSegment& s : segs)
                {
                    next_parents.push_back(ParentRange{cur, cur + s.length});
                    cur += s.length;
                }
            }

            // Decide Regular vs Ragged.
            //   Regular: all parents have the same child count AND share the
            //            same child coordinate values (a Cartesian product).
            //   Ragged:  sizes = the per-parent child counts.
            bool all_equal_count = true;
            bool all_shared_values = true;
            for (std::size_t i = 1; i < child_counts.size(); ++i)
            {
                if (child_counts[i] != child_counts[0])
                    all_equal_count = false;
                if (compact_by_parent[i] != compact_by_parent[0])
                    all_shared_values = false;
            }

            if (all_equal_count && all_shared_values)
            {
                dims.push_back(DimensionSpec::Regular(child_counts[0]));
                // Regular: coordinate values are shared across parents, so the
                // compact axis holds the (single) shared set once.
                axis_data.push_back(
                    DataSeries::CreateFromMeasurements(compact_by_parent[0]));
            }
            else
            {
                dims.push_back(DimensionSpec::Ragged(child_counts));
                // Ragged: each parent has its own coordinate values; the
                // compact axis concatenates them in parent order.
                std::vector<Measurement> compact;
                std::size_t n = 0;
                for (const auto& c : compact_by_parent)
                    n += c.size();
                compact.reserve(n);
                for (const auto& c : compact_by_parent)
                    compact.insert(compact.end(), c.begin(), c.end());
                axis_data.push_back(DataSeries::CreateFromMeasurements(compact));
            }

            parents = std::move(next_parents);
        }

        // Assemble the returned specs.
        std::vector<IndependentSpec> result;
        result.reserve(rank);
        for (std::size_t d = 0; d < rank; ++d)
            result.push_back(IndependentSpec{column_names[d], std::move(axis_data[d]), dims[d]});
        return result;
    }

    void Block::ensure_unique_name(const std::string& name) const
    {
        if (name.empty())
            throw std::invalid_argument("DataArray name must not be empty");
        // DataArray names may be dotted (e.g. "SRC1.i" -- the REL reference
        // syntax supports two-segment dependent names).  They must not
        // contain '/' because HDF5 uses '/' as its own hierarchy separator
        // and a dataset name containing '/' is invalid in HDF5.
        if (name.find('/') != std::string::npos)
            throw std::invalid_argument(
                "DataArray name must not contain '/': " + name);
        if (independent_spec_map_.find(name) != independent_spec_map_.end())
            throw std::invalid_argument("duplicate DataArray name in block: " + name);
        if (dependent_spec_map_.find(name) != dependent_spec_map_.end())
            throw std::invalid_argument("duplicate DataArray name in block: " + name);
    }

    Block::Block(const BlockCreateInfo& info)
    {
        MultiDimensionSpec dependent_multi_dim;   // builds up from all independents in order

        for (const auto& iv : info.independent_specs)
        {
            ensure_unique_name(iv.name);

            independent_spec_map_.emplace(iv.name, iv);

            // Add in order -- add_dimension validates ragged against prior dims.
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
    }

    Block::Block(std::string name, const BlockCreateInfo& info)
        : Block(info)
    {
        if (!IsValidBlockName(name))
            throw std::invalid_argument(
                "Block name must be a valid path: " + name);
        name_ = std::move(name);
    }

    Block::Block(std::string name, BlockCreateInfo&& info)
        : Block(std::move(info))
    {
        if (!IsValidBlockName(name))
            throw std::invalid_argument(
                "Block name must be a valid path: " + name);
        name_ = std::move(name);
    }

    const std::string& Block::name() const
    {
        return name_;
    }

    void Block::set_name(std::string name)
    {
        if (!IsValidBlockName(name))
            throw std::invalid_argument(
                "Block name must be a valid path: " + name);
        name_ = std::move(name);
    }

    void Block::set_source_path(std::string path)
    {
        source_path_ = std::move(path);
    }

    void Block::set_dataset_name(std::string name)
    {
        dataset_name_ = std::move(name);
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
        vinfo.kind = DataArrayKind::kIndependent;

        MultiDimensionSpec composed_multi_dim;
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
            // Prior independents stored raw (no expansion needed).
            vinfo.datas.emplace(kv.first, iv.data);
        }

        if (!found)
            throw std::invalid_argument("independent DataArray not found in block ordering: " + info.name);

        // Self data stored raw (kSelf always last).
        vinfo.datas.emplace(DataArray::kSelf, info.data);
        vinfo.multi_dimension_spec = composed_multi_dim;
        return DataArray(std::move(vinfo));
    }

    const DataArray& Block::GetOrCreateDataArray(const std::string& name) const
    {
        auto cached_it = data_array_cache_.find(name);
        if (cached_it != data_array_cache_.end())
            return *cached_it->second;

        // Try independent first.
        auto it = independent_spec_map_.find(name);
        if (it != independent_spec_map_.end())
        {
            std::unique_ptr<DataArray> var(new DataArray(CreateDataArray(it->second)));
            var->set_source(source_path_, name);
            auto& ref = *var;
            data_array_cache_[name] = std::move(var);
            return ref;
        }

        // Dependent.
        auto dit = dependent_spec_map_.find(name);
        if (dit != dependent_spec_map_.end())
        {
            DataArrayCreateInfo vinfo;
            vinfo.kind = DataArrayKind::kDependent;

            MultiDimensionSpec multi_dim;
            for (const auto& kv : independent_spec_map_)
            {
                const DimensionSpec& dim = kv.second.dimension;
                multi_dim.add_dimension(dim);
                vinfo.datas.emplace(kv.first, kv.second.data);
            }
            vinfo.datas.emplace(DataArray::kSelf, dit->second.data);
            vinfo.multi_dimension_spec = multi_dim;

            std::unique_ptr<DataArray> var(new DataArray(std::move(vinfo)));
            var->set_source(source_path_, name);
            auto& ref = *var;
            data_array_cache_[name] = std::move(var);
            return ref;
        }

        throw std::invalid_argument("DataArray not found: " + name);
    }

    const DataFrame& Block::GetOrCreateDataFrame() const
    {
        if (!data_frame_cache_)
            data_frame_cache_ = DataFrame::FromBlock(*this);
        return *data_frame_cache_;
    }
} // namespace xdataset
