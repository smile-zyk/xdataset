#include "data_array.h"
#include "data_series.h"
#include "operation.h"
#include "dimension_spec.h"
#include "multi_dimension_spec.h"

#include <functional>
#include <map>
#include <stdexcept>
#include <tsl/ordered_map.h>
#include <vector>

namespace xdataset
{

    const char* DataArray::kSelf = "";

    namespace
    {

        /// Shared validation logic (assumes datas are already canonicalized).
        void validate_datas_internal(const tsl::ordered_map<std::string, DataSeries>& datas,
                                     const MultiDimensionSpec&                    multi_dimension_spec,
                                     DataArrayKind                                kind)
        {
            if (datas.empty())
                throw std::invalid_argument("DataArray: datas must not be empty");

            if (datas.rbegin()->first != DataArray::kSelf)
                throw std::invalid_argument(
                    "DataArray: last entry of datas must have key kSelf (empty string)");

            const std::size_t rank = multi_dimension_spec.rank();

            if (kind == DataArrayKind::kIndependent)
            {
                if (datas.size() != rank)
                    throw std::invalid_argument(
                        "DataArray: Independent datas count " + std::to_string(datas.size()) +
                        " must equal multi_dimension_spec rank " + std::to_string(rank));
            }
            else
            {
                if (datas.size() != rank + 1)
                    throw std::invalid_argument(
                        "DataArray: Dependent datas count " + std::to_string(datas.size()) +
                        " must equal rank + 1 (" + std::to_string(rank + 1) + ")");

                if (!multi_dimension_spec.empty())
                {
                    const std::size_t expected = multi_dimension_spec.compute_cell_count();
                    if (datas.rbegin()->second.size() != static_cast<Index>(expected))
                    {
                        throw std::invalid_argument(
                            "DataArray: dependent data size " +
                            std::to_string(datas.rbegin()->second.size()) +
                            " does not match multi_dimension_spec cell count " +
                            std::to_string(expected));
                    }
                }
            }
        }

    } // anonymous namespace

    void DataArray::Validate(const DataArrayCreateInfo& info)
    {
        validate_datas_internal(info.datas, info.multi_dimension_spec, info.kind);
    }

    DataArray::DataArray(const DataArrayCreateInfo& info)
        : datas_(info.datas),
          multi_dimension_spec_(info.multi_dimension_spec),
          data_kind_(info.kind)
    {
        // Canonicalize then validate.  validate() canonicalizes its own copy
        // of the datas, so the member is handled independently.
        for (auto it = datas_.begin(); it != datas_.end(); ++it)
            it.value().canonicalize();
        validate_datas_internal(datas_, multi_dimension_spec_, data_kind_);
    }

    DataArray::DataArray(DataArrayCreateInfo&& info)
        : datas_(std::move(info.datas)),
          multi_dimension_spec_(std::move(info.multi_dimension_spec)),
          data_kind_(info.kind)
    {
        // info.datas has been moved; canonicalize and validate datas_ directly.
        for (auto it = datas_.begin(); it != datas_.end(); ++it)
            it.value().canonicalize();
        validate_datas_internal(datas_, multi_dimension_spec_, data_kind_);
    }

    DataArray::DataArray(const DataArray& other)
        : datas_(other.datas_),
          multi_dimension_spec_(other.multi_dimension_spec_),
          data_kind_(other.data_kind_)
    {
        // data_frame_cache_ intentionally left as nullptr.
    }

    DataArray& DataArray::operator=(const DataArray& other)
    {
        if (this != &other)
        {
            datas_ = other.datas_;
            multi_dimension_spec_ = other.multi_dimension_spec_;
            data_kind_ = other.data_kind_;
            data_frame_cache_.reset();
        }
        return *this;
    }

    tsl::ordered_map<std::string, DataSeries> DataArray::indep_datas() const
    {
        tsl::ordered_map<std::string, DataSeries> result;
        const std::size_t rank = multi_dimension_spec_.rank();
        std::size_t i = 0;
        for (const auto& item : datas_)
        {
            // Dependent: stop after rank entries (exclude kSelf).
            if (data_kind_ == DataArrayKind::kDependent && i >= rank)
                break;
            result.emplace(item.first, item.second);
            ++i;
        }
        return result;
    }

    std::vector<std::string> DataArray::indep_names() const
    {
        std::vector<std::string> names;
        const std::size_t rank = multi_dimension_spec_.rank();
        names.reserve(rank);
        for (const auto& item : datas_)
        {
            if (item.first == kSelf)
                break;  // kSelf is never an independent variable name
            names.push_back(item.first);
        }
        return names;
    }

    const DataFrame& DataArray::GetOrCreateDataFrame(const std::string& variable_name) const
    {
        if (!data_frame_cache_)
        {
            data_frame_cache_.reset(new DataArrayDataFrame(*this, variable_name));
        }
        else
        {
            auto* arr_df = static_cast<DataArrayDataFrame*>(data_frame_cache_.get());
            if (arr_df->variable_name() != variable_name)
                arr_df->UpdateName(variable_name);
        }
        return *data_frame_cache_;
    }

    const DataSeries& DataArray::indep_data(Index index) const
    {
        if (index <= 0)
            throw std::invalid_argument("indep_data index must be 1-based and greater than 0");

        const std::size_t rank = multi_dimension_spec_.rank();
        if (static_cast<std::size_t>(index) > rank)
            throw std::out_of_range("indep_data index out of range");

        // index=1 -> last indep entry, index=rank -> first indep entry
        const std::size_t target = rank - static_cast<std::size_t>(index);
        auto it = datas_.begin();
        std::advance(it, static_cast<std::ptrdiff_t>(target));
        return it->second;
    }

    const DataSeries& DataArray::indep_data(const std::string& name) const
    {
        if (data_kind_ == DataArrayKind::kDependent && name == kSelf)
            throw std::invalid_argument(
                "indep_data: kSelf is not an independent variable for Dependent DataArray");

        auto it = datas_.find(name);
        if (it == datas_.end())
            throw std::invalid_argument("indep_data name not found: " + name);

        // For Dependent, verify the entry is within the first rank entries.
        if (data_kind_ == DataArrayKind::kDependent)
        {
            const std::size_t rank = multi_dimension_spec_.rank();
            std::size_t pos = 0;
            for (auto dit = datas_.begin(); dit != datas_.end(); ++dit, ++pos)
            {
                if (dit->first == name)
                    break;
            }
            if (pos >= rank)
                throw std::invalid_argument(
                    "indep_data: '" + name + "' is not an independent variable");
        }

        return it->second;
    }

    DataArray DataArray::indep(Index index) const
    {
        if (index <= 0)
            throw std::invalid_argument("indep index must be 1-based and greater than 0");

        const std::size_t rank = multi_dimension_spec_.rank();
        if (static_cast<std::size_t>(index) > rank)
            throw std::out_of_range("indep index out of range");

        // index=1 -> last indep entry, index=rank -> first indep entry
        const std::size_t target = rank - static_cast<std::size_t>(index);

        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kIndependent;

        // Copy first (target+1) entries from datas_ �?raw dimension data, no
        // expansion needed.  The last copied entry becomes kSelf.  When the
        // source is Independent and the last entry is the self-dimension,
        // generate an index series (0, 1, ...) instead of copying the data.
        std::size_t pos = 0;
        for (const auto& item : datas_)
        {
            if (pos > target)
                break;

            const bool is_self_entry = (pos == target);
            const bool generate_index =
                is_self_entry && (data_kind_ == DataArrayKind::kIndependent);

            if (generate_index)
            {
                const DataSeries& raw = item.second;
                DataSeries idx =
                    DataSeries::CreateScalar<int>(raw.size(), Unit(), 0);
                for (Index i = 0; i < raw.size(); ++i)
                    idx.scalar_at<int>(i) = static_cast<int>(i);
                info.datas.emplace(kSelf, std::move(idx));
            }
            else if (is_self_entry)
                info.datas.emplace(kSelf, item.second);
            else
                info.datas.emplace(item.first, item.second);
            ++pos;
        }

        // Build result multi_dimension_spec from prefix dimensions.
        MultiDimensionSpec result_spec;
        for (std::size_t i = 0; i <= target; ++i)
            result_spec.add_dimension(multi_dimension_spec_.dims()[i]);
        info.multi_dimension_spec = result_spec;

        return DataArray(std::move(info));
    }

    DataArray DataArray::indep(const std::string& name) const
    {
        if (name.empty())
            throw std::invalid_argument("indep name must not be empty");

        std::size_t pos = 0;
        for (const auto& item : datas_)
        {
            // For Dependent, stop at rank (exclude kSelf).
            if (data_kind_ == DataArrayKind::kDependent && pos >= multi_dimension_spec_.rank())
                break;

            if (item.first == name)
            {
                const std::size_t rank = multi_dimension_spec_.rank();
                const Index index_1_based = static_cast<Index>(rank - pos);
                return indep(index_1_based);
            }
            ++pos;
        }

        throw std::invalid_argument("indep name not found: " + name);
    }

    DataArray DataArray::at(const std::vector<MultiIndexSelector>& selectors) const
    {
        if (data().data_kind() == DataKind::kScalar)
        {
            throw std::logic_error("at is invalid for scalar data");
        }

        const std::size_t ndim = data().data_shape().size();  // 1 for vector, 2 for matrix
        if (selectors.size() > ndim)
        {
            throw std::invalid_argument("too many selectors for at");
        }

        // Pad short selectors with Any so callers can omit trailing dimensions.
        std::vector<MultiIndexSelector> padded = selectors;
        while (padded.size() < ndim)
            padded.push_back(MultiIndexSelector::Any());

        DataArrayCreateInfo info;
        info.kind = data_kind_;
        info.datas = datas_;
        info.multi_dimension_spec = multi_dimension_spec_;

        if (data().data_kind() == DataKind::kVector)
        {
            const std::vector<Index> selected = padded[0].resolve(data().data_shape()[0]);
            info.datas[kSelf] = data().at(selected);
        }
        else
        {
            const std::vector<Index> selected_rows = padded[0].resolve(data().data_shape()[0]);
            const std::vector<Index> selected_cols = padded[1].resolve(data().data_shape()[1]);
            info.datas[kSelf] = data().at(selected_rows, selected_cols);
        }
        return DataArray(std::move(info));
    }

    DataArray DataArray::select(
        const std::vector<MultiIndexSelector>& selectors) const
    {
        const std::size_t rank = multi_dimension_spec_.rank();
        if (rank == 0)
        {
            throw std::logic_error("select requires non-empty dimensions");
        }

        if (selectors.size() > rank)
        {
            throw std::invalid_argument("selector count exceeds DataArray rank");
        }

        // `selectors` may be shorter than rank: normalize it by prepending Any and
        // keeping the user-provided selectors at the end, so short selectors only
        // constrain the trailing dimensions by default.
        std::vector<MultiIndexSelector> actual_selectors;
        actual_selectors.reserve(rank);
        if (selectors.size() < rank)
        {
            actual_selectors.insert(
                actual_selectors.end(), rank - selectors.size(), MultiIndexSelector::Any());
        }
        actual_selectors.insert(actual_selectors.end(), selectors.begin(), selectors.end());

        std::vector<bool> is_dim_retain(rank, true);
        for (std::size_t source_dim = 0; source_dim < rank; ++source_dim)
        {
            is_dim_retain[source_dim] = !actual_selectors[source_dim].is_equal();
        }

        // Independent DataArray: the last dimension (self) must never be eliminated,
        // even when the selector is Equal �?otherwise the result has no data.
        if (data_kind_ == DataArrayKind::kIndependent && rank > 0)
            is_dim_retain[rank - 1] = true;

        struct SelectionDimensionInformation
        {
            bool is_ragged = false;
            std::vector<Index> source_rows;
            std::vector<std::size_t> child_counts;
        };

        // key is source dimension index, value is SelectionDimensionInformation
        std::map<Index, SelectionDimensionInformation> selection_info;

        // for dependent DataArray, record source data selected row
        std::vector<Index> selected_row_indices;

        std::function<void(Index, Index)> walk = [&](Index dim_idx, Index parent_flat)
        {
            if (dim_idx == static_cast<Index>(rank))
            {
                selected_row_indices.push_back(parent_flat);
                return;
            }

            const DimensionSpec& dim = multi_dimension_spec_.dim(dim_idx);
            std::size_t width = 0;
            if (dim.is_regular())
            {
                width = dim.regular_size();
            }
            else
            {
                width = dim.child_width(parent_flat);
            }

            const std::vector<Index> selected_children =
                actual_selectors[static_cast<std::size_t>(dim_idx)].resolve(static_cast<Index>(width));

            selection_info[dim_idx].child_counts.push_back(selected_children.size());
            selection_info[dim_idx].is_ragged = !dim.is_regular();

            if (selection_info[dim_idx].is_ragged)
            {
                for (Index child : selected_children)
                {
                    Index start = 0;
                    Index end = 0;
                    dim.child_range(parent_flat, start, end);
                    Index source_row = start + child;
                    selection_info[dim_idx].source_rows.push_back(source_row);
                }
            }
            else
            {
                if (selection_info[dim_idx].source_rows.empty())
                {
                    selection_info[dim_idx].source_rows = selected_children;
                }
            }

            for (Index child : selected_children)
            {
                Index current_flat = 0;
                if (dim.is_regular())
                {
                    const Index size = static_cast<Index>(dim.regular_size());
                    current_flat = parent_flat * size + child;
                }
                else
                {
                    Index start = 0;
                    Index end = 0;
                    dim.child_range(parent_flat, start, end);
                    (void)end;
                    current_flat = start + child;
                }

                walk(dim_idx + 1, current_flat);
            }
        };

        walk(0, 0);

        MultiDimensionSpec selected_multi_dim;
        for (std::size_t i = 0; i < rank; ++i)
        {
            if (!is_dim_retain[i])
            {
                continue;
            }

            const std::vector<std::size_t>& counts = selection_info[static_cast<Index>(i)].child_counts;
            if (counts.empty())
            {
                selected_multi_dim.add_regular(0);
                continue;
            }

            if (selection_info[static_cast<Index>(i)].is_ragged)
            {
                if (counts.size() == 1)
                {
                    selected_multi_dim.add_regular(counts.front());
                }
                else
                {
                    selected_multi_dim.add_ragged(counts);
                }
            }
            else
            {
                selected_multi_dim.add_regular(counts.front());
            }
        }

        DataArrayCreateInfo info;
        info.kind = data_kind_;
        info.multi_dimension_spec = selected_multi_dim;

        // Iterate datas_ in order and select from each entry.
        // Independent: each position maps 1:1 to a dimension.
        // Dependent:   first rank positions are indep dims; last (kSelf) is
        //              the expanded dependent data, selected by flat index.
        std::size_t idx = 0;
        for (auto it = datas_.begin(); it != datas_.end(); ++it, ++idx)
        {
            const bool is_self = (idx == datas_.size() - 1);   // kSelf entry

            if (is_self && data_kind_ == DataArrayKind::kDependent)
            {
                // Dependent kSelf: select by flat row indices.
                DataSeries sel(data().data_type(), data().data_shape());
                for (Index r : selected_row_indices)
                    sel.append_from(data(), r);
                info.datas.emplace(kSelf, std::move(sel));
            }
            else
            {
                // Dimension data (Independent all entries, Dependent first rank).
                if (!is_dim_retain[idx])
                    continue;

                const auto& src_rows = selection_info[static_cast<Index>(idx)].source_rows;
                const DataSeries& src_series = it->second;
                DataSeries sel(src_series.data_type(), src_series.data_shape());
                for (Index r : src_rows)
                    sel.append_from(src_series, r);

                std::string key = is_self ? std::string(kSelf) : it->first;
                info.datas.emplace(std::move(key), std::move(sel));
            }
        }

        // If only kSelf remains and the original was Dependent, demote to
        // Independent because a Dependent DataArray requires independent
        // variables as dependencies.
        if (data_kind_ == DataArrayKind::kDependent && info.datas.size() == 1)
        {
            info.kind = DataArrayKind::kIndependent;
            const Index data_size = info.datas[kSelf].size();
            info.multi_dimension_spec =
                MultiDimensionSpec().add_regular(static_cast<std::size_t>(data_size));
        }

        return DataArray(std::move(info));
    }

    // =========================================================================
    //  Innermost-dimension reduction (min / max)
    // =========================================================================

    namespace
    {
        /// Three-way comparison used by min/max. Numeric values compare
        /// numerically, complex by magnitude (std::abs), booleans as
        /// false < true, and strings lexicographically.
        int compare_values(const Measurement& a, const Measurement& b)
        {
            switch (a.data_type())
            {
                case DataType::kReal:
                {
                    const double x = boost::get<double>(a.storage());
                    const double y = boost::get<double>(b.storage());
                    return x < y ? -1 : (x > y ? 1 : 0);
                }
                case DataType::kInteger:
                {
                    const int x = boost::get<int>(a.storage());
                    const int y = boost::get<int>(b.storage());
                    return x < y ? -1 : (x > y ? 1 : 0);
                }
                case DataType::kComplex:
                {
                    const double x = std::abs(boost::get<std::complex<double>>(a.storage()));
                    const double y = std::abs(boost::get<std::complex<double>>(b.storage()));
                    return x < y ? -1 : (x > y ? 1 : 0);
                }
                case DataType::kBoolean:
                {
                    const bool x = boost::get<bool>(a.storage());
                    const bool y = boost::get<bool>(b.storage());
                    return x == y ? 0 : (x ? 1 : -1);
                }
                case DataType::kString:
                {
                    const std::string& x = boost::get<std::string>(a.storage());
                    const std::string& y = boost::get<std::string>(b.storage());
                    return x < y ? -1 : (x > y ? 1 : 0);
                }
            }
            return 0;
        }
    } // anonymous namespace

    DataArray DataArray::min() const { return reduce_minmax(false); }
    DataArray DataArray::max() const { return reduce_minmax(true); }

    DataArray DataArray::reduce_minmax(bool want_max) const
    {
        if (data().data_kind() != DataKind::kScalar)
        {
            throw std::logic_error(
                want_max ? "max is only supported for scalar data"
                         : "min is only supported for scalar data");
        }

        const std::size_t rank = multi_dimension_spec_.rank();
        if (rank == 0)
        {
            throw std::logic_error("min/max requires at least one dimension");
        }

        const auto better = [want_max](const Measurement& a, const Measurement& b) -> bool
        {
            // True when `a` replaces `b` as the running extreme.
            const int cmp = compare_values(a, b);
            return want_max ? (cmp > 0) : (cmp < 0);
        };

        struct GroupAcc
        {
            Measurement best;
            bool        have = false;
        };

        std::vector<GroupAcc> groups;

        // The innermost dimension (rank - 1) is the reduction axis.  Its
        // "outer prefix" is dims [0 .. rank - 2], so for_each_group_at_dim
        // must be called with (rank - 2).  For rank == 1 the whole column is
        // a single group.
        if (rank >= 2)
        {
            multi_dimension_spec_.for_each_group_at_dim(
                static_cast<Index>(rank - 2),
                [&](const MultiDimensionSpec::DimGroup& g)
                {
                    GroupAcc acc;
                    multi_dimension_spec_.for_each_leaf_row(
                        [&](const MultiDimensionSpec::LeafRow& leaf)
                        {
                            // Dependent: self data is expanded, index by flat row.
                            // Independent: self is the raw innermost dimension,
                            // index by the innermost dimension row.
                            const Measurement val =
                                (data_kind_ == DataArrayKind::kIndependent)
                                    ? data().measurement_at(
                                          leaf.dimension_row_indices[rank - 1])
                                    : data().measurement_at(leaf.row_flat);
                            if (!acc.have)
                            {
                                acc.best = val;
                                acc.have = true;
                            }
                            else if (better(val, acc.best))
                            {
                                acc.best = val;
                            }
                        },
                        g.flat_start, g.flat_end);
                    groups.push_back(std::move(acc));
                });
        }
        else
        {
            GroupAcc acc;
            const Index total =
                static_cast<Index>(multi_dimension_spec_.compute_cell_count());
            for (Index f = 0; f < total; ++f)
            {
                const Measurement val = data().measurement_at(f);
                if (!acc.have)
                {
                    acc.best = val;
                    acc.have = true;
                }
                else if (better(val, acc.best))
                {
                    acc.best = val;
                }
            }
            groups.push_back(std::move(acc));
        }

        // Reduced values become the new kSelf data (one row per group).
        DataSeries out(data().data_type(), data().data_shape());
        for (const GroupAcc& acc : groups)
            out.append(acc.best);

        DataArrayCreateInfo info;

        if (rank >= 2)
        {
            // Remaining dimensions survive: the reduced values become
            // dependent data over the remaining independent dimensions,
            // so the result is always Dependent (even if the input was
            // Independent — the input's own values are now the data).
            info.kind = DataArrayKind::kDependent;
            auto it = datas_.begin();
            for (std::size_t i = 0; i < rank - 1; ++i, ++it)
                info.datas[it->first] = it->second;
            info.datas[kSelf] = std::move(out);   // kSelf must stay last

            MultiDimensionSpec spec;
            for (std::size_t i = 0; i < rank - 1; ++i)
                spec.add_dimension(multi_dimension_spec_.dim(static_cast<Index>(i)));
            info.multi_dimension_spec = std::move(spec);
        }
        else
        {
            // No dimensions left: demote to a single-value Independent.
            info.kind = DataArrayKind::kIndependent;
            info.datas[kSelf] = std::move(out);
            info.multi_dimension_spec = MultiDimensionSpec().add_regular(1);
        }

        return DataArray(std::move(info));
    }

    // Static factory methods

    DataArray DataArray::CreateIndependent(
        DataSeries data)
    {
        const std::size_t size = data.size();
        DataArrayCreateInfo vinfo;
        vinfo.datas[kSelf] = std::move(data);
        vinfo.multi_dimension_spec = MultiDimensionSpec().add_regular(size);
        vinfo.kind = DataArrayKind::kIndependent;
        return DataArray(std::move(vinfo));
    }

    DataArray DataArray::CreateDependent(
        DataSeries data,
        const tsl::ordered_map<std::string, const DataArray*>& indep_variables)
    {
        if (indep_variables.empty())
        {
            throw std::invalid_argument(
                "CreateDependent: indep_variables must not be empty");
        }

        MultiDimensionSpec spec;

        for (const auto& item : indep_variables)
        {
            const std::string& var_name = item.first;
            const DataArray*   var      = item.second;
            if (!var)
            {
                throw std::invalid_argument(
                    "CreateDependent: null indep_variable in list");
            }
            if (var->data_kind() != DataArrayKind::kIndependent)
            {
                throw std::invalid_argument(
                    "CreateDependent: DataArray is not an independent DataArray");
            }

            const std::vector<DimensionSpec>& dims = var->multi_dimension_spec().dims();
            if (dims.empty())
            {
                throw std::logic_error(
                    "CreateDependent: independent DataArray has no dimensions");
            }
            spec.add_dimension(dims.back());   // validates parent count for ragged
        }

        DataArrayCreateInfo vinfo;
        // Collect indep variable data first (ordered by insertion into spec).
        for (const auto& item : indep_variables)
        {
            vinfo.datas[item.first] = item.second->data();
        }
        // Self data (dependent) goes last.
        vinfo.datas[kSelf] = std::move(data);
        vinfo.multi_dimension_spec = std::move(spec);
        vinfo.kind = DataArrayKind::kDependent;
        return DataArray(std::move(vinfo));
    }

// =========================================================================
//  DataArray -- replace_self_data / with_self_data
// =========================================================================

void DataArray::replace_self_data(DataSeries&& new_self)
{
    // For Dependent, validate that the new series size matches the cell count.
    if (data_kind_ == DataArrayKind::kDependent && !multi_dimension_spec_.empty())
    {
        const std::size_t expected = multi_dimension_spec_.compute_cell_count();
        if (new_self.size() != static_cast<Index>(expected))
        {
            throw std::invalid_argument(
                "replace_self_data: new series size " +
                std::to_string(new_self.size()) +
                " does not match multi_dimension_spec cell count " +
                std::to_string(expected));
        }
    }

    // Canonicalize the new series for consistency with DataArray invariants.
    new_self.canonicalize();

    // Replace the last entry (kSelf) in the ordered map.
    datas_[kSelf] = std::move(new_self);

    // Invalidate cached DataFrame.
    data_frame_cache_.reset();
}

void DataArray::replace_self_data(const DataSeries& new_self)
{
    replace_self_data(DataSeries(new_self));
}

DataArray DataArray::with_self_data(DataSeries&& new_self) const
{
    DataArray result(*this);
    result.replace_self_data(std::move(new_self));
    return result;
}

DataArray DataArray::with_self_data(const DataSeries& new_self) const
{
    DataArray result(*this);
    result.replace_self_data(DataSeries(new_self));
    return result;
}

// =========================================================================
DataArray operator+(const DataArray& a, const DataArray& b)  { return OperationAdd(Value(a),Value(b)).as_data_array(); }
DataArray operator-(const DataArray& a, const DataArray& b)  { return OperationSub(Value(a),Value(b)).as_data_array(); }
DataArray operator*(const DataArray& a, const DataArray& b)  { return OperationMul(Value(a),Value(b)).as_data_array(); }
DataArray operator/(const DataArray& a, const DataArray& b)  { return OperationDiv(Value(a),Value(b)).as_data_array(); }
DataArray operator+(const DataArray& a, const Measurement& b){ return OperationAdd(Value(a),Value(b)).as_data_array(); }
DataArray operator-(const DataArray& a, const Measurement& b){ return OperationSub(Value(a),Value(b)).as_data_array(); }
DataArray operator*(const DataArray& a, const Measurement& b){ return OperationMul(Value(a),Value(b)).as_data_array(); }
DataArray operator/(const DataArray& a, const Measurement& b){ return OperationDiv(Value(a),Value(b)).as_data_array(); }
DataArray operator+(const Measurement& a, const DataArray& b){ return OperationAdd(Value(a),Value(b)).as_data_array(); }
DataArray operator-(const Measurement& a, const DataArray& b){ return OperationSub(Value(a),Value(b)).as_data_array(); }
DataArray operator*(const Measurement& a, const DataArray& b){ return OperationMul(Value(a),Value(b)).as_data_array(); }
DataArray operator/(const Measurement& a, const DataArray& b){ return OperationDiv(Value(a),Value(b)).as_data_array(); }

// -- comparison (AA, AM, MA)
DataArray operator==(const DataArray& a, const DataArray& b) { return OperationEq(Value(a),Value(b)).as_data_array(); }
DataArray operator!=(const DataArray& a, const DataArray& b) { return OperationNeq(Value(a),Value(b)).as_data_array(); }
DataArray operator<(const DataArray& a, const DataArray& b)  { return OperationLt(Value(a),Value(b)).as_data_array(); }
DataArray operator>(const DataArray& a, const DataArray& b)  { return OperationGt(Value(a),Value(b)).as_data_array(); }
DataArray operator<=(const DataArray& a, const DataArray& b) { return OperationLe(Value(a),Value(b)).as_data_array(); }
DataArray operator>=(const DataArray& a, const DataArray& b) { return OperationGe(Value(a),Value(b)).as_data_array(); }

DataArray operator==(const DataArray& a, const Measurement& b) { return OperationEq(Value(a),Value(b)).as_data_array(); }
DataArray operator!=(const DataArray& a, const Measurement& b) { return OperationNeq(Value(a),Value(b)).as_data_array(); }
DataArray operator<(const DataArray& a, const Measurement& b)  { return OperationLt(Value(a),Value(b)).as_data_array(); }
DataArray operator>(const DataArray& a, const Measurement& b)  { return OperationGt(Value(a),Value(b)).as_data_array(); }
DataArray operator<=(const DataArray& a, const Measurement& b) { return OperationLe(Value(a),Value(b)).as_data_array(); }
DataArray operator>=(const DataArray& a, const Measurement& b) { return OperationGe(Value(a),Value(b)).as_data_array(); }

DataArray operator==(const Measurement& a, const DataArray& b) { return OperationEq(Value(a),Value(b)).as_data_array(); }
DataArray operator!=(const Measurement& a, const DataArray& b) { return OperationNeq(Value(a),Value(b)).as_data_array(); }
DataArray operator<(const Measurement& a, const DataArray& b)  { return OperationLt(Value(a),Value(b)).as_data_array(); }
DataArray operator>(const Measurement& a, const DataArray& b)  { return OperationGt(Value(a),Value(b)).as_data_array(); }
DataArray operator<=(const Measurement& a, const DataArray& b) { return OperationLe(Value(a),Value(b)).as_data_array(); }
DataArray operator>=(const Measurement& a, const DataArray& b) { return OperationGe(Value(a),Value(b)).as_data_array(); }

// -- logical (AA, AM, MA)
DataArray operator&&(const DataArray& a, const DataArray& b){ return OperationAnd(Value(a),Value(b)).as_data_array(); }
DataArray operator||(const DataArray& a, const DataArray& b){ return OperationOr(Value(a),Value(b)).as_data_array(); }
DataArray operator&&(const DataArray& a, const Measurement& b){ return OperationAnd(Value(a),Value(b)).as_data_array(); }
DataArray operator||(const DataArray& a, const Measurement& b){ return OperationOr(Value(a),Value(b)).as_data_array(); }
DataArray operator&&(const Measurement& a, const DataArray& b){ return OperationAnd(Value(a),Value(b)).as_data_array(); }
DataArray operator||(const Measurement& a, const DataArray& b){ return OperationOr(Value(a),Value(b)).as_data_array(); }

// -- bitwise (AA, AM, MA)
DataArray operator&(const DataArray& a, const DataArray& b) { return OperationBitAnd(Value(a),Value(b)).as_data_array(); }
DataArray operator|(const DataArray& a, const DataArray& b) { return OperationBitOr(Value(a),Value(b)).as_data_array(); }
DataArray operator^(const DataArray& a, const DataArray& b) { return OperationBitXor(Value(a),Value(b)).as_data_array(); }
DataArray operator&(const DataArray& a, const Measurement& b){ return OperationBitAnd(Value(a),Value(b)).as_data_array(); }
DataArray operator|(const DataArray& a, const Measurement& b){ return OperationBitOr(Value(a),Value(b)).as_data_array(); }
DataArray operator^(const DataArray& a, const Measurement& b){ return OperationBitXor(Value(a),Value(b)).as_data_array(); }
DataArray operator&(const Measurement& a, const DataArray& b){ return OperationBitAnd(Value(a),Value(b)).as_data_array(); }
DataArray operator|(const Measurement& a, const DataArray& b){ return OperationBitOr(Value(a),Value(b)).as_data_array(); }
DataArray operator^(const Measurement& a, const DataArray& b){ return OperationBitXor(Value(a),Value(b)).as_data_array(); }

// -- shift (AA, AM, MA)
DataArray operator<<(const DataArray& a, const DataArray& b){ return OperationShl(Value(a),Value(b)).as_data_array(); }
DataArray operator>>(const DataArray& a, const DataArray& b){ return OperationShr(Value(a),Value(b)).as_data_array(); }
DataArray operator<<(const DataArray& a, const Measurement& b){ return OperationShl(Value(a),Value(b)).as_data_array(); }
DataArray operator>>(const DataArray& a, const Measurement& b){ return OperationShr(Value(a),Value(b)).as_data_array(); }
DataArray operator<<(const Measurement& a, const DataArray& b){ return OperationShl(Value(a),Value(b)).as_data_array(); }
DataArray operator>>(const Measurement& a, const DataArray& b){ return OperationShr(Value(a),Value(b)).as_data_array(); }

// -- modulo (AA, AM, MA)
DataArray operator%(const DataArray& a, const DataArray& b) { return OperationMod(Value(a),Value(b)).as_data_array(); }
DataArray operator%(const DataArray& a, const Measurement& b){ return OperationMod(Value(a),Value(b)).as_data_array(); }
DataArray operator%(const Measurement& a, const DataArray& b){ return OperationMod(Value(a),Value(b)).as_data_array(); }

// -- unary
DataArray operator-(const DataArray& v) { return OperationNegate(Value(v)).as_data_array(); }
DataArray operator!(const DataArray& v) { return OperationNot(Value(v)).as_data_array(); }
DataArray operator~(const DataArray& v) { return OperationBitNot(Value(v)).as_data_array(); }

// -- pow
DataArray pow(const DataArray& base, const DataArray& exp)      { return OperationPow(Value(base),Value(exp)).as_data_array(); }
DataArray pow(const DataArray& base, const Measurement& exp)    { return OperationPow(Value(base),Value(exp)).as_data_array(); }
DataArray pow(const Measurement& base, const DataArray& exp)    { return OperationPow(Value(base),Value(exp)).as_data_array(); }

} // namespace xdataset
