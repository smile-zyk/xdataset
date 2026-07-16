#include "grid_model.h"
#include "block.h"
#include "cell_series.h"
#include "variable.h"

#include <complex>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace xdataset
{
    namespace
    {
        std::string EscapeCsvCell(const std::string& cell)
        {
            bool need_quote = false;
            for (char ch : cell)
            {
                if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r')
                {
                    need_quote = true;
                    break;
                }
            }

            if (!need_quote)
            {
                return cell;
            }

            std::string escaped;
            escaped.reserve(cell.size() + 2);
            escaped.push_back('"');
            for (char ch : cell)
            {
                if (ch == '"')
                {
                    escaped.push_back('"');
                }
                escaped.push_back(ch);
            }
            escaped.push_back('"');
            return escaped;
        }
    } // namespace

    // =========================================================================
    // GridFieldStringifyVisitor (singleton)
    // =========================================================================

    const GridFieldStringifyVisitor& GridFieldStringifyVisitor::Instance()
    {
        static const GridFieldStringifyVisitor instance;
        return instance;
    }

    std::string GridFieldStringifyVisitor::operator()(double v) const
    {
        std::ostringstream oss;
        oss << v;
        return oss.str();
    }

    std::string GridFieldStringifyVisitor::operator()(int v) const
    {
        std::ostringstream oss;
        oss << v;
        return oss.str();
    }

    std::string GridFieldStringifyVisitor::operator()(const std::complex<double>& v) const
    {
        std::ostringstream oss;
        oss << v.real();
        if (v.imag() >= 0.0) oss << "+";
        oss << v.imag() << "i";
        return oss.str();
    }

    std::string GridFieldStringifyVisitor::operator()(const std::string& v) const
    {
        return v;
    }

    // =========================================================================
    // GridFieldGetVisitor (singleton)
    // =========================================================================

    const GridFieldGetVisitor& GridFieldGetVisitor::Apply(const GridField& field)
    {
        static GridFieldGetVisitor instance;
        boost::apply_visitor(instance, field);
        return instance;
    }

    DTypeTag GridFieldGetVisitor::TypeOf(const GridField& field)
    {
        GridFieldGetVisitor v;
        boost::apply_visitor(v, field);
        return v.type_;
    }

    bool GridFieldGetVisitor::IsDouble(const GridField& field) { return TypeOf(field) == DTypeTag::kReal; }
    bool GridFieldGetVisitor::IsInt(const GridField& field)    { return TypeOf(field) == DTypeTag::kInteger; }
    bool GridFieldGetVisitor::IsComplex(const GridField& field){ return TypeOf(field) == DTypeTag::kComplex; }
    bool GridFieldGetVisitor::IsString(const GridField& field) { return TypeOf(field) == DTypeTag::kString; }

    void GridFieldGetVisitor::operator()(double v)
    {
        type_  = DTypeTag::kReal;
        d_val_ = v;
    }

    void GridFieldGetVisitor::operator()(int v)
    {
        type_  = DTypeTag::kInteger;
        i_val_ = v;
    }

    void GridFieldGetVisitor::operator()(const std::complex<double>& v)
    {
        type_  = DTypeTag::kComplex;
        c_val_ = v;
    }

    void GridFieldGetVisitor::operator()(const std::string& v)
    {
        type_  = DTypeTag::kString;
        s_val_ = v;
    }

    // =========================================================================
    // GridRow
    // =========================================================================

    std::string GridRow::FormatMultiIndex() const
    {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < multi_index.size(); ++i)
        {
            if (i > 0) oss << ",";
            oss << multi_index[i];
        }
        oss << "]";
        return oss.str();
    }

    // =========================================================================
    // Internal helpers (file-local)
    // =========================================================================

    static std::vector<std::string> ExpandHeadersForSeries(
        const std::string& base_name,
        const CellSeries& series)
    {
        std::vector<std::string> headers;
        if (series.cell_kind() == CellKind::kScalar)
        {
            headers.push_back(base_name);
            return headers;
        }

        if (series.cell_kind() == CellKind::kVector)
        {
            const Index width = series.cell_shape()[0];
            headers.reserve(static_cast<std::size_t>(width));
            for (Index i = 0; i < width; ++i)
            {
                headers.push_back(base_name + "(" + std::to_string(static_cast<std::size_t>(i) + 1) + ")");
            }
            return headers;
        }

        if (series.cell_kind() == CellKind::kMatrix)
        {
            const Index rows = series.cell_shape()[0];
            const Index cols = series.cell_shape()[1];
            headers.reserve(static_cast<std::size_t>(rows * cols));
            for (Index r = 0; r < rows; ++r)
            {
                for (Index c = 0; c < cols; ++c)
                {
                    headers.push_back(
                        base_name +
                        "(" + std::to_string(static_cast<std::size_t>(r) + 1) +
                        "," + std::to_string(static_cast<std::size_t>(c) + 1) +
                        ")");
                }
            }
            return headers;
        }

        throw std::logic_error("unsupported cell kind");
    }

    static std::vector<GridField> CellAtToColumns(const CellSeries& series, Index row)
    {
        std::vector<GridField> cells;

        if (series.cell_kind() == CellKind::kScalar)
        {
            switch (series.dtype())
            {
                case DTypeTag::kReal:    cells.push_back(GridField(series.scalar_at<double>(row)));              return cells;
                case DTypeTag::kInteger: cells.push_back(GridField(series.scalar_at<int>(row)));                 return cells;
                case DTypeTag::kComplex: cells.push_back(GridField(series.scalar_at<std::complex<double>>(row))); return cells;
                case DTypeTag::kString:  cells.push_back(GridField(series.scalar_at<std::string>(row)));         return cells;
            }
            throw std::logic_error("unsupported scalar dtype");
        }

        if (series.cell_kind() == CellKind::kVector)
        {
            const Index width = series.cell_shape()[0];
            cells.reserve(static_cast<std::size_t>(width));
            switch (series.dtype())
            {
                case DTypeTag::kReal:
                {
                    const auto vec = series.vector_at<double>(row);
                    for (Index i = 0; i < width; ++i) cells.push_back(GridField(vec(i)));
                    return cells;
                }
                case DTypeTag::kInteger:
                {
                    const auto vec = series.vector_at<int>(row);
                    for (Index i = 0; i < width; ++i) cells.push_back(GridField(vec(i)));
                    return cells;
                }
                case DTypeTag::kComplex:
                {
                    const auto vec = series.vector_at<std::complex<double>>(row);
                    for (Index i = 0; i < width; ++i) cells.push_back(GridField(vec(i)));
                    return cells;
                }
                case DTypeTag::kString:
                {
                    const auto& vec = series.vector_at<std::string>(row);
                    for (Index i = 0; i < width; ++i) cells.push_back(GridField(vec(i)));
                    return cells;
                }
            }
            throw std::logic_error("unsupported vector dtype");
        }

        if (series.cell_kind() == CellKind::kMatrix)
        {
            const Index rows = series.cell_shape()[0];
            const Index cols = series.cell_shape()[1];
            cells.reserve(static_cast<std::size_t>(rows * cols));
            switch (series.dtype())
            {
                case DTypeTag::kReal:
                {
                    const auto mat = series.matrix_at<double>(row);
                    for (Index r = 0; r < rows; ++r)
                        for (Index c = 0; c < cols; ++c)
                            cells.push_back(GridField(mat(r, c)));
                    return cells;
                }
                case DTypeTag::kInteger:
                {
                    const auto mat = series.matrix_at<int>(row);
                    for (Index r = 0; r < rows; ++r)
                        for (Index c = 0; c < cols; ++c)
                            cells.push_back(GridField(mat(r, c)));
                    return cells;
                }
                case DTypeTag::kComplex:
                {
                    const auto mat = series.matrix_at<std::complex<double>>(row);
                    for (Index r = 0; r < rows; ++r)
                        for (Index c = 0; c < cols; ++c)
                            cells.push_back(GridField(mat(r, c)));
                    return cells;
                }
                case DTypeTag::kString:
                {
                    const auto& mat = series.matrix_at<std::string>(row);
                    for (Index r = 0; r < rows; ++r)
                        for (Index c = 0; c < cols; ++c)
                            cells.push_back(GridField(mat(r, c)));
                    return cells;
                }
            }
            throw std::logic_error("unsupported matrix dtype");
        }

        throw std::logic_error("unsupported cell kind");
    }

    static std::string CellAtToString(const CellSeries& series, Index row)
    {
        const std::vector<GridField> cols = CellAtToColumns(series, row);
        if (cols.empty())
        {
            return {};
        }
        if (cols.size() == 1)
        {
            return GridFieldToString(cols[0]);
        }

        std::ostringstream oss;
        oss << "[";

        if (series.cell_kind() == CellKind::kMatrix)
        {
            const Index mat_rows = series.cell_shape()[0];
            const Index mat_cols = series.cell_shape()[1];
            for (Index r = 0; r < mat_rows; ++r)
            {
                if (r > 0) oss << ",";
                oss << "[";
                for (Index c = 0; c < mat_cols; ++c)
                {
                    if (c > 0) oss << ",";
                    oss << GridFieldToString(cols[static_cast<std::size_t>(r * mat_cols + c)]);
                }
                oss << "]";
            }
        }
        else
        {
            for (std::size_t i = 0; i < cols.size(); ++i)
            {
                if (i > 0) oss << ",";
                oss << GridFieldToString(cols[i]);
            }
        }

        oss << "]";
        return oss.str();
    }

    // =========================================================================
    // GridModel — lazy loading
    // =========================================================================

    void GridModel::Configure(std::vector<std::string> headers,
                              std::size_t total_rows,
                              RowGenerator generator,
                              std::size_t chunk_size)
    {
        headers_     = std::move(headers);
        total_rows_  = total_rows;
        chunk_size_  = (chunk_size > 0) ? chunk_size : 256;
        generator_   = std::move(generator);

        // Pre-allocate cache slots so operator[] is safe for any valid row.
        rows_.resize(total_rows_);
        loaded_chunks_.assign((total_rows_ + chunk_size_ - 1) / chunk_size_, false);
    }

    const GridRow& GridModel::GetRow(Index row) const
    {
        EnsureChunkLoaded(row / static_cast<Index>(chunk_size_));
        return rows_[static_cast<std::size_t>(row)];
    }

    void GridModel::EnsureChunkLoaded(Index chunk_idx) const
    {
        const std::size_t ci = static_cast<std::size_t>(chunk_idx);
        if (loaded_chunks_[ci])
        {
            return;
        }

        const Index start = static_cast<Index>(ci * chunk_size_);
        const Index end   = static_cast<Index>((std::min)((ci + 1) * chunk_size_, total_rows_));

        std::vector<GridRow> chunk_rows = generator_(start, end);

        for (std::size_t i = 0; i < chunk_rows.size(); ++i)
        {
            rows_[static_cast<std::size_t>(start) + i] = std::move(chunk_rows[i]);
        }

        loaded_chunks_[ci] = true;
    }

    // =========================================================================
    // WriteToCsv / ToCsv
    // =========================================================================

    std::string GridModel::ToCsv() const
    {
        std::string csv;

        // Header row: leading comma for multi-index column
        for (const auto& header : headers_)
        {
            csv.push_back(',');
            csv += EscapeCsvCell(header);
        }
        csv.push_back('\n');

        // Data rows (lazy — triggers chunk loads on demand)
        for (std::size_t i = 0; i < total_rows_; ++i)
        {
            const GridRow& row = GetRow(static_cast<Index>(i));
            csv += EscapeCsvCell(row.FormatMultiIndex());

            for (const GridField& field : row.fields)
            {
                csv.push_back(',');
                csv += EscapeCsvCell(GridFieldToString(field));
            }
            csv.push_back('\n');
        }

        return csv;
    }

    void GridModel::WriteToCsv(const std::string& file_path) const
    {
        if (file_path.empty())
        {
            throw std::invalid_argument("csv file path must not be empty");
        }

        std::ofstream ofs(file_path.c_str(), std::ios::out | std::ios::trunc);
        if (!ofs.is_open())
        {
            throw std::runtime_error("failed to open csv file: " + file_path);
        }

        ofs << ToCsv();

        if (!ofs.good())
        {
            throw std::runtime_error("failed to write csv file: " + file_path);
        }
    }

    // =========================================================================
    // BlockGridModel
    // =========================================================================

    BlockGridModel::BlockGridModel(const Block& block)
    {
        std::vector<std::string> all_headers;
        std::vector<DimensionSpec> dims;
        std::vector<std::string> indep_names = block.independents();
        dims.reserve(indep_names.size());

        for (const std::string& indep_name : indep_names)
        {
            const IndependentVariableInfo& iv = block.independent_variable(indep_name);
            const std::vector<std::string> hdrs = ExpandHeadersForSeries(indep_name, iv.data);
            all_headers.insert(all_headers.end(), hdrs.begin(), hdrs.end());
            dims.push_back(iv.dimension);
        }

        std::vector<std::string> dep_names = block.dependents();
        for (const std::string& dep_name : dep_names)
        {
            const DependentVariableInfo& dv = block.dependent_variable(dep_name);
            const std::vector<std::string> hdrs = ExpandHeadersForSeries(dep_name, dv.data);
            all_headers.insert(all_headers.end(), hdrs.begin(), hdrs.end());
        }

        const MultiDimensionSpec traversal_spec(dims);
        const std::size_t total_rows = dims.empty() ? 0 : traversal_spec.compute_cell_count();
        const std::size_t total_headers = all_headers.size();
        const Block*      b = &block;

        Configure(std::move(all_headers), total_rows,
            [b, traversal_spec, total_headers, indep_names, dep_names](
                Index start, Index end) -> std::vector<GridRow>
            {
                std::vector<GridRow> result;
                result.reserve(static_cast<std::size_t>(end - start));
                traversal_spec.for_each_leaf_row(
                    [&](const MultiDimensionSpec::LeafRow& leaf_row)
                    {
                        GridRow row;
                        row.multi_index = leaf_row.multi_index;
                        row.fields.reserve(total_headers);

                        const auto& dim_ri = leaf_row.dimension_row_indices;
                        for (std::size_t d = 0; d < indep_names.size(); ++d)
                        {
                            const IndependentVariableInfo& iv = b->independent_variable(indep_names[d]);
                            const std::vector<GridField> vals = CellAtToColumns(iv.data, dim_ri[d]);
                            row.fields.insert(row.fields.end(), vals.begin(), vals.end());
                        }
                        for (const std::string& dn : dep_names)
                        {
                            const DependentVariableInfo& dv = b->dependent_variable(dn);
                            const std::vector<GridField> vals = CellAtToColumns(dv.data, leaf_row.row_flat);
                            row.fields.insert(row.fields.end(), vals.begin(), vals.end());
                        }
                        result.push_back(std::move(row));
                    },
                    start, end);
                return result;
            });
    }

    // =========================================================================
    // VariableGridModel
    // =========================================================================

    VariableGridModel::VariableGridModel(const Variable& variable)
    {
        std::vector<std::pair<std::string, const CellSeries*>> indep_columns;
        std::vector<std::string> all_headers;

        indep_columns.reserve(variable.indep_datas().size());
        for (const auto& item : variable.indep_datas())
        {
            indep_columns.push_back(std::make_pair(item.first, &item.second));
            const std::vector<std::string> hdrs = ExpandHeadersForSeries(item.first, item.second);
            all_headers.insert(all_headers.end(), hdrs.begin(), hdrs.end());
        }

        const CellSeries* dep_series = nullptr;
        if (variable.kind() == VariableKind::kDependent)
        {
            dep_series = &variable.data();
            const std::vector<std::string> hdrs = ExpandHeadersForSeries("data", variable.data());
            all_headers.insert(all_headers.end(), hdrs.begin(), hdrs.end());
        }

        const MultiDimensionSpec& spec = variable.multi_dimension_spec();
        if (spec.empty())
            throw std::logic_error("variable table view requires non-empty dimensions");
        if (indep_columns.size() != spec.rank())
            throw std::logic_error("independent columns count must match MultiDimensionSpec rank");

        const std::size_t total_rows   = spec.compute_cell_count();
        const std::size_t total_headers = all_headers.size();

        Configure(std::move(all_headers), total_rows,
            [spec, indep_columns, total_headers, dep_series](
                Index start, Index end) -> std::vector<GridRow>
            {
                std::vector<GridRow> result;
                result.reserve(static_cast<std::size_t>(end - start));
                spec.for_each_leaf_row(
                    [&](const MultiDimensionSpec::LeafRow& leaf_row)
                    {
                        GridRow row;
                        row.multi_index = leaf_row.multi_index;
                        row.fields.reserve(total_headers);

                        const auto& dim_ri = leaf_row.dimension_row_indices;
                        for (std::size_t d = 0; d < indep_columns.size(); ++d)
                        {
                            const CellSeries* is = indep_columns[d].second;
                            if (dim_ri[d] < 0 || static_cast<std::size_t>(dim_ri[d]) >= is->size())
                                throw std::out_of_range("expanded independent row index out of bounds");
                            const std::vector<GridField> vals = CellAtToColumns(*is, dim_ri[d]);
                            row.fields.insert(row.fields.end(), vals.begin(), vals.end());
                        }
                        if (dep_series != nullptr)
                        {
                            if (leaf_row.row_flat < 0 || static_cast<std::size_t>(leaf_row.row_flat) >= dep_series->size())
                                throw std::out_of_range("dependent data row index out of bounds");
                            const std::vector<GridField> vals = CellAtToColumns(*dep_series, leaf_row.row_flat);
                            row.fields.insert(row.fields.end(), vals.begin(), vals.end());
                        }
                        result.push_back(std::move(row));
                    },
                    start, end);
                return result;
            });
    }
} // namespace xdataset
