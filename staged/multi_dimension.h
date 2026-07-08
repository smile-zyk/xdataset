#ifndef XDATASET_MULTI_DIMENSION_H_
#define XDATASET_MULTI_DIMENSION_H_

#include "cell_series.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xdataset {

class DimensionSpec {
public:
    enum Kind {
        kUniform,
        kJagged
    };

    static DimensionSpec Uniform(Index size) {
        if (size < 0) {
            throw std::invalid_argument("uniform dimension size must be non-negative");
        }
        return DimensionSpec(kUniform, size, std::vector<Index>());
    }

    static DimensionSpec Jagged(const std::vector<Index>& sizes) {
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] < 0) {
                throw std::invalid_argument("jagged dimension size must be non-negative");
            }
        }
        return DimensionSpec(kJagged, 0, sizes);
    }

    Kind kind() const { return kind_; }
    bool is_uniform() const { return kind_ == kUniform; }
    bool is_jagged() const { return kind_ == kJagged; }

    Index uniform_size() const {
        if (!is_uniform()) {
            throw std::logic_error("uniform_size() is only valid for uniform dimensions");
        }
        return uniform_size_;
    }

    const std::vector<Index>& jagged_sizes() const {
        if (!is_jagged()) {
            throw std::logic_error("jagged_sizes() is only valid for jagged dimensions");
        }
        return jagged_sizes_;
    }

private:
    DimensionSpec(Kind kind, Index uniform_size, const std::vector<Index>& jagged_sizes)
        : kind_(kind), uniform_size_(uniform_size), jagged_sizes_(jagged_sizes) {}

    Kind kind_;
    Index uniform_size_;
    std::vector<Index> jagged_sizes_;
};

class MultiIndexSelector {
public:
    enum Kind {
        kAny,
        kEqual,
        kIn
    };

    static MultiIndexSelector Any() {
        return MultiIndexSelector(kAny, std::vector<Index>());
    }

    static MultiIndexSelector Equal(Index idx) {
        if (idx == -1) {
            return Any();
        }
        if (idx < 0) {
            throw std::invalid_argument("equal selector index must be >= 0 or -1");
        }
        return MultiIndexSelector(kEqual, std::vector<Index>(1, idx));
    }

    static MultiIndexSelector In(const std::vector<Index>& indices) {
        if (indices.empty()) {
            throw std::invalid_argument("in selector must not be empty");
        }

        bool has_any = false;
        std::vector<Index> normalized;
        normalized.reserve(indices.size());

        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] == -1) {
                has_any = true;
                break;
            }
            if (indices[i] < 0) {
                throw std::invalid_argument("in selector index must be >= 0 or -1");
            }
            normalized.push_back(indices[i]);
        }

        if (has_any) {
            return Any();
        }

        std::sort(normalized.begin(), normalized.end());
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
        return MultiIndexSelector(kIn, normalized);
    }

    Kind kind() const { return kind_; }
    bool is_any() const { return kind_ == kAny; }
    bool is_equal() const { return kind_ == kEqual; }
    bool is_in() const { return kind_ == kIn; }

    Index equal_value() const {
        if (!is_equal()) {
            throw std::logic_error("equal_value() is only valid for equal selectors");
        }
        return values_[0];
    }

    const std::vector<Index>& in_values() const {
        if (!is_in()) {
            throw std::logic_error("in_values() is only valid for in selectors");
        }
        return values_;
    }

    bool matches(Index idx) const {
        if (kind_ == kAny) {
            return true;
        }
        if (kind_ == kEqual) {
            return idx == values_[0];
        }
        return std::binary_search(values_.begin(), values_.end(), idx);
    }

private:
    MultiIndexSelector(Kind kind, const std::vector<Index>& values)
        : kind_(kind), values_(values) {}

    Kind kind_;
    std::vector<Index> values_;
};

struct MultiIndexLevelLayout {
    std::vector<std::size_t> child_counts;
    std::vector<std::size_t> child_prefixes;
};

class MultiDimensionSpec {
public:
    struct ProjectedSelectionResult {
        std::vector<std::size_t> flat_indices;
        std::vector<std::vector<Index> > projected_multi_indices;
    };

    MultiDimensionSpec()
        : cache_valid_(false), cache_has_jagged_(false) {}

    explicit MultiDimensionSpec(const std::vector<DimensionSpec>& dims)
        : dims_(dims), cache_valid_(false), cache_has_jagged_(false) {}

    static MultiDimensionSpec UniformShape(const std::vector<Index>& sizes) {
        MultiDimensionSpec spec;
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            spec.add_uniform(sizes[i]);
        }
        return spec;
    }

    MultiDimensionSpec& add_uniform(Index size) {
        dims_.push_back(DimensionSpec::Uniform(size));
        invalidate_cache();
        return *this;
    }

    MultiDimensionSpec& add_jagged(const std::vector<Index>& sizes) {
        dims_.push_back(DimensionSpec::Jagged(sizes));
        invalidate_cache();
        return *this;
    }

    MultiDimensionSpec& add_dimension(const DimensionSpec& dim) {
        dims_.push_back(dim);
        invalidate_cache();
        return *this;
    }

    std::size_t rank() const { return dims_.size(); }
    std::size_t ndim() const { return rank(); }
    bool empty() const { return dims_.empty(); }
    const std::vector<DimensionSpec>& dims() const { return dims_; }

    std::vector<std::size_t> level_node_counts() const {
        ensure_cache();
        return cached_level_counts_;
    }

    std::size_t total_cell_count() const {
        ensure_cache();
        return cached_level_counts_.back();
    }

    std::size_t total_size() const { return total_cell_count(); }

    std::vector<MultiIndexLevelLayout> build_layout() const {
        ensure_cache();
        return cached_layout_;
    }

    std::vector<std::vector<std::size_t> > innermost_groups() const {
        std::vector<std::vector<std::size_t> > groups;
        if (dims_.empty()) {
            return groups;
        }

        ensure_cache();
        const MultiIndexLevelLayout& last_level = cached_layout_.back();
        groups.resize(last_level.child_counts.size());

        for (std::size_t p = 0; p < last_level.child_counts.size(); ++p) {
            const std::size_t begin = last_level.child_prefixes[p];
            const std::size_t end = last_level.child_prefixes[p + 1];
            groups[p].reserve(end - begin);
            for (std::size_t flat = begin; flat < end; ++flat) {
                groups[p].push_back(flat);
            }
        }

        return groups;
    }

    void validate_definition() const {
        (void)level_node_counts();
    }

    void validate_exact_multi_index(const std::vector<Index>& exact_index) const {
        if (exact_index.size() != dims_.size()) {
            throw std::invalid_argument("number of indices must match rank");
        }
        if (dims_.empty()) {
            return;
        }

        ensure_cache();

        if (!cache_has_jagged_) {
            for (std::size_t d = 0; d < dims_.size(); ++d) {
                const Index idx = exact_index[d];
                if (idx < 0) {
                    throw std::out_of_range("exact index must be non-negative");
                }
                if (static_cast<std::size_t>(idx) >= cached_uniform_sizes_[d]) {
                    throw std::out_of_range("index out of bounds in dimension");
                }
            }
            return;
        }

        std::size_t parent_index = 0;
        for (std::size_t d = 0; d < dims_.size(); ++d) {
            const Index idx = exact_index[d];
            if (idx < 0) {
                throw std::out_of_range("exact index must be non-negative");
            }
            const std::size_t child_count = cached_layout_[d].child_counts[parent_index];
            if (static_cast<std::size_t>(idx) >= child_count) {
                throw std::out_of_range("index out of bounds in dimension");
            }
            parent_index = cached_layout_[d].child_prefixes[parent_index] +
                           static_cast<std::size_t>(idx);
        }
    }

    std::size_t flat_index(const std::vector<Index>& exact_index) const {
        validate_exact_multi_index(exact_index);
        if (dims_.empty()) {
            return 0;
        }

        ensure_cache();
        if (!cache_has_jagged_) {
            std::size_t flat = 0;
            for (std::size_t d = 0; d < dims_.size(); ++d) {
                flat = checked_add(
                    flat,
                    checked_mul(static_cast<std::size_t>(exact_index[d]),
                                cached_uniform_strides_[d],
                                "uniform flat index overflow"),
                    "uniform flat index overflow");
            }
            return flat;
        }

        std::size_t parent_index = 0;
        for (std::size_t d = 0; d < dims_.size(); ++d) {
            parent_index = cached_layout_[d].child_prefixes[parent_index] +
                           static_cast<std::size_t>(exact_index[d]);
        }
        return parent_index;
    }

    std::size_t at(const std::vector<Index>& exact_index) const { return flat_index(exact_index); }

    std::vector<Index> multi_index(std::size_t flat) const {
        if (dims_.empty()) {
            if (flat != 0) {
                throw std::out_of_range("flat index out of bounds");
            }
            return std::vector<Index>();
        }

        ensure_cache();
        const std::size_t total = cached_level_counts_.back();
        if (flat >= total) {
            throw std::out_of_range("flat index out of bounds");
        }

        std::vector<Index> out(dims_.size(), 0);

        if (!cache_has_jagged_) {
            std::size_t residual = flat;
            for (std::size_t d = 0; d < dims_.size(); ++d) {
                const std::size_t stride = cached_uniform_strides_[d];
                const std::size_t idx = residual / stride;
                out[d] = static_cast<Index>(idx);
                residual -= idx * stride;
            }
            return out;
        }

        std::size_t node_index = flat;

        for (std::size_t d = dims_.size(); d-- > 0;) {
            const std::vector<std::size_t>& prefixes = cached_layout_[d].child_prefixes;
            const std::vector<std::size_t>::const_iterator it =
                std::upper_bound(prefixes.begin(), prefixes.end(), node_index);
            const std::size_t parent_index = static_cast<std::size_t>((it - prefixes.begin()) - 1);
            const std::size_t child_index = node_index - prefixes[parent_index];
            out[d] = static_cast<Index>(child_index);
            node_index = parent_index;
        }

        return out;
    }

    std::vector<std::size_t> flat_indices(const std::vector<MultiIndexSelector>& selectors) const {
        if (selectors.size() != dims_.size()) {
            throw std::invalid_argument("number of selectors must match rank");
        }

        if (dims_.empty()) {
            return std::vector<std::size_t>(1, 0);
        }

        ensure_cache();
        std::vector<std::size_t> out;
        if (!cache_has_jagged_) {
            out.reserve(estimate_uniform_hits(selectors));
            collect_uniform_hits(selectors, 0, 0, &out);
            return out;
        }

        collect_jagged_hits(selectors, 0, 0, &out);
        return out;
    }

    std::vector<std::size_t> select_flat_indices(
        const std::vector<MultiIndexSelector>& selectors) const {
        return flat_indices(selectors);
    }

    std::vector<std::size_t> flat_indices(const std::vector<Index>& selectors_with_any) const {
        if (selectors_with_any.size() != dims_.size()) {
            throw std::invalid_argument("number of selectors must match rank");
        }
        std::vector<MultiIndexSelector> selectors;
        selectors.reserve(selectors_with_any.size());
        for (std::size_t d = 0; d < selectors_with_any.size(); ++d) {
            selectors.push_back(MultiIndexSelector::Equal(selectors_with_any[d]));
        }
        return flat_indices(selectors);
    }

    std::vector<std::size_t> select_flat_indices(
        const std::vector<Index>& selectors_with_any) const {
        return flat_indices(selectors_with_any);
    }

    ProjectedSelectionResult select_with_projection(
        const std::vector<MultiIndexSelector>& selectors,
        const std::vector<std::size_t>& retained_dims) const {
        if (selectors.size() != dims_.size()) {
            throw std::invalid_argument("number of selectors must match rank");
        }

        const std::vector<Index> projected_dim_positions =
            build_projected_dim_positions(selectors.size(), retained_dims);

        ProjectedSelectionResult out;
        if (dims_.empty()) {
            out.flat_indices.push_back(0);
            out.projected_multi_indices.push_back(std::vector<Index>());
            return out;
        }

        ensure_cache();
        if (!cache_has_jagged_) {
            reserve_projected_result_for_uniform(selectors, &out);
        }

        // The current DFS path's projected coordinates. At each leaf we snapshot it
        // into out.projected_multi_indices alongside the matched flat index.
        std::vector<Index> projected_index(retained_dims.size(), 0);
        if (!cache_has_jagged_) {
            collect_uniform_hits_with_projection(
                selectors,
                projected_dim_positions,
                0,
                0,
                &projected_index,
                &out);
            return out;
        }

        collect_jagged_hits_with_projection(
            selectors,
            projected_dim_positions,
            0,
            0,
            &projected_index,
            &out);
        return out;
    }

private:
    static std::size_t checked_add(std::size_t left,
                                   std::size_t right,
                                   const char* context_message) {
        if (left > (std::numeric_limits<std::size_t>::max() - right)) {
            throw std::overflow_error(context_message);
        }
        return left + right;
    }

    static std::size_t checked_mul(std::size_t left,
                                   std::size_t right,
                                   const char* context_message) {
        if (left == 0 || right == 0) {
            return 0;
        }
        if (left > (std::numeric_limits<std::size_t>::max() / right)) {
            throw std::overflow_error(context_message);
        }
        return left * right;
    }

    static void write_projected_coordinate(Index projected_slot,
                                           Index value,
                                           std::vector<Index>* projected_index) {
        if (projected_slot >= 0) {
            (*projected_index)[static_cast<std::size_t>(projected_slot)] = value;
        }
    }

    static std::vector<Index> build_projected_dim_positions(
        std::size_t selector_count,
        const std::vector<std::size_t>& retained_dims) {
        std::vector<Index> projected_dim_positions(selector_count, Index(-1));
        for (std::size_t i = 0; i < retained_dims.size(); ++i) {
            const std::size_t dim = retained_dims[i];
            if (dim >= selector_count) {
                throw std::invalid_argument("retained dimension out of range");
            }
            if (projected_dim_positions[dim] != Index(-1)) {
                throw std::invalid_argument("retained dimensions must be unique");
            }
            projected_dim_positions[dim] = static_cast<Index>(i);
        }
        return projected_dim_positions;
    }

    void reserve_projected_result_for_uniform(
        const std::vector<MultiIndexSelector>& selectors,
        ProjectedSelectionResult* out) const {
        const std::size_t estimate = estimate_uniform_hits(selectors);
        out->flat_indices.reserve(estimate);
        out->projected_multi_indices.reserve(estimate);
    }

    void invalidate_cache() {
        cache_valid_ = false;
        cache_has_jagged_ = false;
        cached_level_counts_.clear();
        cached_layout_.clear();
        cached_uniform_sizes_.clear();
        cached_uniform_strides_.clear();
    }

    void ensure_cache() const {
        if (cache_valid_) {
            return;
        }

        std::vector<std::size_t> level_counts;
        level_counts.reserve(dims_.size() + 1);
        std::vector<MultiIndexLevelLayout> layout;
        layout.reserve(dims_.size());
        std::vector<std::size_t> uniform_sizes;
        uniform_sizes.reserve(dims_.size());
        bool has_jagged = false;

        std::size_t parent_count = 1;
        level_counts.push_back(parent_count);

        for (std::size_t d = 0; d < dims_.size(); ++d) {
            const DimensionSpec& dim = dims_[d];
            MultiIndexLevelLayout level;
            level.child_counts.resize(parent_count, 0);

            if (dim.is_uniform()) {
                const std::size_t count = static_cast<std::size_t>(dim.uniform_size());
                uniform_sizes.push_back(count);
                for (std::size_t p = 0; p < parent_count; ++p) {
                    level.child_counts[p] = count;
                }
                parent_count = checked_mul(
                    parent_count,
                    count,
                    "dimension size overflow when building uniform level");
            } else {
                has_jagged = true;
                uniform_sizes.push_back(0);
                const std::vector<Index>& sizes = dim.jagged_sizes();
                if (sizes.size() != parent_count) {
                    throw std::invalid_argument(
                        "jagged dimension size list length must match parent node count");
                }

                std::size_t next_count = 0;
                for (std::size_t p = 0; p < parent_count; ++p) {
                    const std::size_t child_count = static_cast<std::size_t>(sizes[p]);
                    level.child_counts[p] = child_count;
                    next_count = checked_add(
                        next_count,
                        child_count,
                        "dimension size overflow when building jagged level");
                }
                parent_count = next_count;
            }

            level.child_prefixes.resize(level.child_counts.size() + 1, 0);
            for (std::size_t p = 0; p < level.child_counts.size(); ++p) {
                level.child_prefixes[p + 1] = checked_add(
                    level.child_prefixes[p],
                    level.child_counts[p],
                    "dimension prefix overflow when building layout");
            }

            level_counts.push_back(parent_count);
            layout.push_back(level);
        }

        std::vector<std::size_t> uniform_strides;
        if (!has_jagged && !uniform_sizes.empty()) {
            uniform_strides.resize(uniform_sizes.size(), 1);
            for (std::size_t d = uniform_sizes.size(); d-- > 0;) {
                if (d + 1 < uniform_sizes.size()) {
                    uniform_strides[d] = checked_mul(
                        uniform_strides[d + 1],
                        uniform_sizes[d + 1],
                        "uniform stride overflow when building cache");
                }
            }
        }

        cached_level_counts_.swap(level_counts);
        cached_layout_.swap(layout);
        cached_uniform_sizes_.swap(uniform_sizes);
        cached_uniform_strides_.swap(uniform_strides);
        cache_has_jagged_ = has_jagged;
        cache_valid_ = true;
    }

    std::size_t estimate_uniform_hits(
        const std::vector<MultiIndexSelector>& selectors) const {
        std::size_t count = 1;
        for (std::size_t d = 0; d < selectors.size(); ++d) {
            const std::size_t dim_size = cached_uniform_sizes_[d];
            std::size_t selected_count = 0;
            const MultiIndexSelector& selector = selectors[d];

            if (selector.is_any()) {
                selected_count = dim_size;
            } else if (selector.is_equal()) {
                const Index idx = selector.equal_value();
                selected_count = (idx >= 0 && static_cast<std::size_t>(idx) < dim_size) ? 1 : 0;
            } else {
                const std::vector<Index>& values = selector.in_values();
                for (std::size_t i = 0; i < values.size(); ++i) {
                    const Index idx = values[i];
                    if (idx >= 0 && static_cast<std::size_t>(idx) < dim_size) {
                        ++selected_count;
                    }
                }
            }

            if (selected_count == 0) {
                return 0;
            }
            count = checked_mul(
                count,
                selected_count,
                "uniform query result size overflow");
        }
        return count;
    }

    void collect_uniform_hits(
        const std::vector<MultiIndexSelector>& selectors,
        std::size_t dim,
        std::size_t flat_prefix,
        std::vector<std::size_t>* out) const {
        const std::size_t stride = cached_uniform_strides_[dim];
        const std::size_t dim_size = cached_uniform_sizes_[dim];
        const MultiIndexSelector& selector = selectors[dim];
        const bool is_last_dim = (dim + 1 == selectors.size());

        if (selector.is_any()) {
            for (std::size_t idx = 0; idx < dim_size; ++idx) {
                const std::size_t next_flat = checked_add(
                    flat_prefix,
                    checked_mul(idx, stride, "uniform flat index overflow"),
                    "uniform flat index overflow");
                if (is_last_dim) {
                    out->push_back(next_flat);
                } else {
                    collect_uniform_hits(selectors, dim + 1, next_flat, out);
                }
            }
            return;
        }

        if (selector.is_equal()) {
            const Index idx = selector.equal_value();
            if (idx < 0 || static_cast<std::size_t>(idx) >= dim_size) {
                return;
            }

            const std::size_t next_flat = checked_add(
                flat_prefix,
                checked_mul(static_cast<std::size_t>(idx), stride, "uniform flat index overflow"),
                "uniform flat index overflow");
            if (is_last_dim) {
                out->push_back(next_flat);
            } else {
                collect_uniform_hits(selectors, dim + 1, next_flat, out);
            }
            return;
        }

        const std::vector<Index>& values = selector.in_values();
        for (std::size_t i = 0; i < values.size(); ++i) {
            const Index idx = values[i];
            if (idx < 0 || static_cast<std::size_t>(idx) >= dim_size) {
                continue;
            }

            const std::size_t next_flat = checked_add(
                flat_prefix,
                checked_mul(static_cast<std::size_t>(idx), stride, "uniform flat index overflow"),
                "uniform flat index overflow");
            if (is_last_dim) {
                out->push_back(next_flat);
            } else {
                collect_uniform_hits(selectors, dim + 1, next_flat, out);
            }
        }
    }

    void collect_jagged_hits(const std::vector<MultiIndexSelector>& selectors,
                             std::size_t dim,
                             std::size_t parent_index,
                             std::vector<std::size_t>* out) const {
        const MultiIndexLevelLayout& level = cached_layout_[dim];
        const std::size_t child_base = level.child_prefixes[parent_index];
        const std::size_t child_count = level.child_counts[parent_index];
        const MultiIndexSelector& selector = selectors[dim];
        const bool is_last_dim = (dim + 1 == selectors.size());

        if (selector.is_any()) {
            for (std::size_t child = 0; child < child_count; ++child) {
                const std::size_t next_node = child_base + child;
                if (is_last_dim) {
                    out->push_back(next_node);
                } else {
                    collect_jagged_hits(selectors, dim + 1, next_node, out);
                }
            }
            return;
        }

        if (selector.is_equal()) {
            const Index exact = selector.equal_value();
            if (exact < 0 || static_cast<std::size_t>(exact) >= child_count) {
                return;
            }

            const std::size_t next_node = child_base + static_cast<std::size_t>(exact);
            if (is_last_dim) {
                out->push_back(next_node);
            } else {
                collect_jagged_hits(selectors, dim + 1, next_node, out);
            }
            return;
        }

        const std::vector<Index>& one_of = selector.in_values();
        for (std::size_t i = 0; i < one_of.size(); ++i) {
            const Index selected = one_of[i];
            if (selected < 0 || static_cast<std::size_t>(selected) >= child_count) {
                continue;
            }

            const std::size_t next_node = child_base + static_cast<std::size_t>(selected);
            if (is_last_dim) {
                out->push_back(next_node);
            } else {
                collect_jagged_hits(selectors, dim + 1, next_node, out);
            }
        }
    }

    void collect_uniform_hits_with_projection(
        const std::vector<MultiIndexSelector>& selectors,
        const std::vector<Index>& projected_dim_positions,
        std::size_t dim,
        std::size_t flat_prefix,
        std::vector<Index>* projected_index,
        ProjectedSelectionResult* out) const {
        const std::size_t stride = cached_uniform_strides_[dim];
        const std::size_t dim_size = cached_uniform_sizes_[dim];
        const MultiIndexSelector& selector = selectors[dim];
        const bool is_last_dim = (dim + 1 == selectors.size());
        const Index projected_slot = projected_dim_positions[dim];

        if (selector.is_any()) {
            for (std::size_t idx = 0; idx < dim_size; ++idx) {
                write_projected_coordinate(
                    projected_slot,
                    static_cast<Index>(idx),
                    projected_index);

                const std::size_t next_flat = checked_add(
                    flat_prefix,
                    checked_mul(idx, stride, "uniform flat index overflow"),
                    "uniform flat index overflow");
                if (is_last_dim) {
                    out->flat_indices.push_back(next_flat);
                    out->projected_multi_indices.push_back(*projected_index);
                } else {
                    collect_uniform_hits_with_projection(
                        selectors,
                        projected_dim_positions,
                        dim + 1,
                        next_flat,
                        projected_index,
                        out);
                }
            }
            return;
        }

        if (selector.is_equal()) {
            const Index idx = selector.equal_value();
            if (idx < 0 || static_cast<std::size_t>(idx) >= dim_size) {
                return;
            }

            write_projected_coordinate(projected_slot, idx, projected_index);

            const std::size_t next_flat = checked_add(
                flat_prefix,
                checked_mul(static_cast<std::size_t>(idx), stride, "uniform flat index overflow"),
                "uniform flat index overflow");
            if (is_last_dim) {
                out->flat_indices.push_back(next_flat);
                out->projected_multi_indices.push_back(*projected_index);
            } else {
                collect_uniform_hits_with_projection(
                    selectors,
                    projected_dim_positions,
                    dim + 1,
                    next_flat,
                    projected_index,
                    out);
            }
            return;
        }

        const std::vector<Index>& values = selector.in_values();
        for (std::size_t i = 0; i < values.size(); ++i) {
            const Index idx = values[i];
            if (idx < 0 || static_cast<std::size_t>(idx) >= dim_size) {
                continue;
            }

            write_projected_coordinate(projected_slot, static_cast<Index>(i), projected_index);

            const std::size_t next_flat = checked_add(
                flat_prefix,
                checked_mul(static_cast<std::size_t>(idx), stride, "uniform flat index overflow"),
                "uniform flat index overflow");
            if (is_last_dim) {
                out->flat_indices.push_back(next_flat);
                out->projected_multi_indices.push_back(*projected_index);
            } else {
                collect_uniform_hits_with_projection(
                    selectors,
                    projected_dim_positions,
                    dim + 1,
                    next_flat,
                    projected_index,
                    out);
            }
        }
    }

    void collect_jagged_hits_with_projection(
        const std::vector<MultiIndexSelector>& selectors,
        const std::vector<Index>& projected_dim_positions,
        std::size_t dim,
        std::size_t parent_index,
        std::vector<Index>* projected_index,
        ProjectedSelectionResult* out) const {
        const MultiIndexLevelLayout& level = cached_layout_[dim];
        const std::size_t child_base = level.child_prefixes[parent_index];
        const std::size_t child_count = level.child_counts[parent_index];
        const MultiIndexSelector& selector = selectors[dim];
        const bool is_last_dim = (dim + 1 == selectors.size());
        const Index projected_slot = projected_dim_positions[dim];

        if (selector.is_any()) {
            for (std::size_t child = 0; child < child_count; ++child) {
                write_projected_coordinate(
                    projected_slot,
                    static_cast<Index>(child),
                    projected_index);

                const std::size_t next_node = child_base + child;
                if (is_last_dim) {
                    out->flat_indices.push_back(next_node);
                    out->projected_multi_indices.push_back(*projected_index);
                } else {
                    collect_jagged_hits_with_projection(
                        selectors,
                        projected_dim_positions,
                        dim + 1,
                        next_node,
                        projected_index,
                        out);
                }
            }
            return;
        }

        if (selector.is_equal()) {
            const Index exact = selector.equal_value();
            if (exact < 0 || static_cast<std::size_t>(exact) >= child_count) {
                return;
            }

            write_projected_coordinate(projected_slot, exact, projected_index);

            const std::size_t next_node = child_base + static_cast<std::size_t>(exact);
            if (is_last_dim) {
                out->flat_indices.push_back(next_node);
                out->projected_multi_indices.push_back(*projected_index);
            } else {
                collect_jagged_hits_with_projection(
                    selectors,
                    projected_dim_positions,
                    dim + 1,
                    next_node,
                    projected_index,
                    out);
            }
            return;
        }

        const std::vector<Index>& one_of = selector.in_values();
        for (std::size_t i = 0; i < one_of.size(); ++i) {
            const Index selected = one_of[i];
            if (selected < 0 || static_cast<std::size_t>(selected) >= child_count) {
                continue;
            }

            write_projected_coordinate(projected_slot, static_cast<Index>(i), projected_index);

            const std::size_t next_node = child_base + static_cast<std::size_t>(selected);
            if (is_last_dim) {
                out->flat_indices.push_back(next_node);
                out->projected_multi_indices.push_back(*projected_index);
            } else {
                collect_jagged_hits_with_projection(
                    selectors,
                    projected_dim_positions,
                    dim + 1,
                    next_node,
                    projected_index,
                    out);
            }
        }
    }

    std::vector<DimensionSpec> dims_;
    mutable bool cache_valid_;
    mutable bool cache_has_jagged_;
    mutable std::vector<std::size_t> cached_level_counts_;
    mutable std::vector<MultiIndexLevelLayout> cached_layout_;
    mutable std::vector<std::size_t> cached_uniform_sizes_;
    mutable std::vector<std::size_t> cached_uniform_strides_;
};

}  // namespace xdataset

#endif  // XDATASET_MULTI_DIMENSION_H_
