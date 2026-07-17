#include "data_frame.h"
#include "block.h"
#include "data_series.h"
#include "data_array.h"
#include "measurement.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace xdataset
{
    namespace
    {
        std::string EscapeCsvField(const std::string& field)
        {
            bool need_quote = false;
            for (char ch : field)
            {
                if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r')
                {
                    need_quote = true;
                    break;
                }
            }

            if (!need_quote)
            {
                return field;
            }

            std::string escaped;
            escaped.reserve(field.size() + 2);
            escaped.push_back('"');
            for (char ch : field)
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
    // DataFrameRow
    // =========================================================================

    std::string DataFrameRow::FormatMultiIndex() const
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
        const DataSeries& series)
    {
        std::vector<std::string> headers;
        if (series.data_kind() == DataKind::kScalar)
        {
            headers.push_back(base_name);
            return headers;
        }

        if (series.data_kind() == DataKind::kVector)
        {
            const Index width = series.data_shape()[0];
            headers.reserve(static_cast<std::size_t>(width));
            for (Index i = 0; i < width; ++i)
            {
                headers.push_back(base_name + "(" + std::to_string(static_cast<std::size_t>(i) + 1) + ")");
            }
            return headers;
        }

        if (series.data_kind() == DataKind::kMatrix)
        {
            const Index rows = series.data_shape()[0];
            const Index cols = series.data_shape()[1];
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

        throw std::logic_error("unsupported data kind");
    }

    static std::vector<Measurement> ExpandToColumns(const DataSeries& series, Index row)
    {
        Measurement m = series.measurement_at(row);
        std::vector<Measurement> result;

        if (series.data_kind() == DataKind::kScalar)
            return {m};

        if (series.data_kind() == DataKind::kVector)
        {
            const Index width = m.shape()[0];
            result.reserve(static_cast<std::size_t>(width));
            for (Index i = 0; i < width; ++i)
                result.push_back(m.element_at(i));
            return result;
        }

        // Matrix
        const Index rows = m.shape()[0];
        const Index cols = m.shape()[1];
        result.reserve(static_cast<std::size_t>(rows * cols));
        for (Index r = 0; r < rows; ++r)
            for (Index c = 0; c < cols; ++c)
                result.push_back(m.element_at(r, c));
        return result;
    }

    static std::string ExpandToString(const DataSeries& series, Index row)
    {
        const std::vector<Measurement> cols = ExpandToColumns(series, row);
        if (cols.empty())
        {
            return {};
        }
        if (cols.size() == 1)
        {
            return cols[0].to_string();
        }

        std::ostringstream oss;
        oss << "[";

        if (series.data_kind() == DataKind::kMatrix)
        {
            const Index mat_rows = series.data_shape()[0];
            const Index mat_cols = series.data_shape()[1];
            for (Index r = 0; r < mat_rows; ++r)
            {
                if (r > 0) oss << ",";
                oss << "[";
                for (Index c = 0; c < mat_cols; ++c)
                {
                    if (c > 0) oss << ",";
                    oss << cols[static_cast<std::size_t>(r * mat_cols + c)].to_string();
                }
                oss << "]";
            }
        }
        else
        {
            for (std::size_t i = 0; i < cols.size(); ++i)
            {
                if (i > 0) oss << ",";
                oss << cols[i].to_string();
            }
        }

        oss << "]";
        return oss.str();
    }

    // =========================================================================
    // DataFrame: lazy loading
    // =========================================================================

    void DataFrame::Configure(std::vector<std::string> headers,
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

    const DataFrameRow& DataFrame::GetRow(Index row) const
    {
        EnsureChunkLoaded(row / static_cast<Index>(chunk_size_));
        return rows_[static_cast<std::size_t>(row)];
    }

    void DataFrame::EnsureChunkLoaded(Index chunk_idx) const
    {
        const std::size_t ci = static_cast<std::size_t>(chunk_idx);
        if (loaded_chunks_[ci])
        {
            return;
        }

        const Index start = static_cast<Index>(ci * chunk_size_);
        const Index end   = static_cast<Index>((std::min)((ci + 1) * chunk_size_, total_rows_));

        std::vector<DataFrameRow> chunk_rows = generator_(start, end);

        for (std::size_t i = 0; i < chunk_rows.size(); ++i)
        {
            rows_[static_cast<std::size_t>(start) + i] = std::move(chunk_rows[i]);
        }

        loaded_chunks_[ci] = true;
    }

    // =========================================================================
    // WriteToCsv / ToCsv
    // =========================================================================

    std::string DataFrame::ToCsv() const
    {
        std::string csv;

        // Header row: leading comma for multi-index column
        for (const auto& header : headers_)
        {
            csv.push_back(',');
            csv += EscapeCsvField(header);
        }
        csv.push_back('\n');

        // Data rows (lazy -- triggers chunk loads on demand)
        for (std::size_t i = 0; i < total_rows_; ++i)
        {
            const DataFrameRow& row = GetRow(static_cast<Index>(i));
            csv += EscapeCsvField(row.FormatMultiIndex());

            for (const Measurement& field : row.fields)
            {
                csv.push_back(',');
                csv += EscapeCsvField(field.to_string());
            }
            csv.push_back('\n');
        }

        return csv;
    }

    void DataFrame::WriteToCsv(const std::string& file_path) const
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
    // BlockDataFrame
    // =========================================================================

    BlockDataFrame::BlockDataFrame(const Block& block)
    {
        std::vector<std::string> all_headers;
        std::vector<DimensionSpec> dims;
        std::vector<std::string> indep_names = block.independents();
        dims.reserve(indep_names.size());

        for (const std::string& indep_name : indep_names)
        {
            const IndependentSpec& iv = block.independent_spec(indep_name);
            const std::vector<std::string> hdrs = ExpandHeadersForSeries(indep_name, iv.data);
            all_headers.insert(all_headers.end(), hdrs.begin(), hdrs.end());
            dims.push_back(iv.dimension);
        }

        std::vector<std::string> dep_names = block.dependents();
        for (const std::string& dep_name : dep_names)
        {
            const DependentSpec& dv = block.dependent_spec(dep_name);
            const std::vector<std::string> hdrs = ExpandHeadersForSeries(dep_name, dv.data);
            all_headers.insert(all_headers.end(), hdrs.begin(), hdrs.end());
        }

        const MultiDimensionSpec traversal_spec(dims);
        const std::size_t total_rows = dims.empty() ? 0 : traversal_spec.compute_cell_count();
        const std::size_t total_headers = all_headers.size();
        const Block*      b = &block;

        Configure(std::move(all_headers), total_rows,
            [b, traversal_spec, total_headers, indep_names, dep_names](
                Index start, Index end) -> std::vector<DataFrameRow>
            {
                std::vector<DataFrameRow> result;
                result.reserve(static_cast<std::size_t>(end - start));
                traversal_spec.for_each_leaf_row(
                    [&](const MultiDimensionSpec::LeafRow& leaf_row)
                    {
                        DataFrameRow row;
                        row.multi_index = leaf_row.multi_index;
                        row.fields.reserve(total_headers);

                        const auto& dim_ri = leaf_row.dimension_row_indices;
                        for (std::size_t d = 0; d < indep_names.size(); ++d)
                        {
                            const IndependentSpec& iv = b->independent_spec(indep_names[d]);
                            const std::vector<Measurement> vals = ExpandToColumns(iv.data, dim_ri[d]);
                            row.fields.insert(row.fields.end(), vals.begin(), vals.end());
                        }
                        for (const std::string& dn : dep_names)
                        {
                            const DependentSpec& dv = b->dependent_spec(dn);
                            const std::vector<Measurement> vals = ExpandToColumns(dv.data, leaf_row.row_flat);
                            row.fields.insert(row.fields.end(), vals.begin(), vals.end());
                        }
                        result.push_back(std::move(row));
                    },
                    start, end);
                return result;
            });
    }

    // =========================================================================
    // DataArrayDataFrame
    // =========================================================================

    DataArrayDataFrame::DataArrayDataFrame(const DataArray& DataArray)
    {
        std::vector<std::pair<std::string, const DataSeries*>> indep_columns;
        std::vector<std::string> all_headers;

        indep_columns.reserve(DataArray.indep_datas().size());
        for (const auto& item : DataArray.indep_datas())
        {
            indep_columns.push_back(std::make_pair(item.first, &item.second));
            const std::vector<std::string> hdrs = ExpandHeadersForSeries(item.first, item.second);
            all_headers.insert(all_headers.end(), hdrs.begin(), hdrs.end());
        }

        const DataSeries* dep_series = nullptr;
        if (DataArray.kind() == DataArrayKind::kDependent)
        {
            dep_series = &DataArray.data();
            const std::vector<std::string> hdrs = ExpandHeadersForSeries("data", DataArray.data());
            all_headers.insert(all_headers.end(), hdrs.begin(), hdrs.end());
        }

        const MultiDimensionSpec& spec = DataArray.multi_dimension_spec();
        if (spec.empty())
            throw std::logic_error("DataArray table view requires non-empty dimensions");
        if (indep_columns.size() != spec.rank())
            throw std::logic_error("independent columns count must match MultiDimensionSpec rank");

        const std::size_t total_rows   = spec.compute_cell_count();
        const std::size_t total_headers = all_headers.size();

        Configure(std::move(all_headers), total_rows,
            [spec, indep_columns, total_headers, dep_series](
                Index start, Index end) -> std::vector<DataFrameRow>
            {
                std::vector<DataFrameRow> result;
                result.reserve(static_cast<std::size_t>(end - start));
                spec.for_each_leaf_row(
                    [&](const MultiDimensionSpec::LeafRow& leaf_row)
                    {
                        DataFrameRow row;
                        row.multi_index = leaf_row.multi_index;
                        row.fields.reserve(total_headers);

                        const auto& dim_ri = leaf_row.dimension_row_indices;
                        for (std::size_t d = 0; d < indep_columns.size(); ++d)
                        {
                            const DataSeries* is = indep_columns[d].second;
                            if (dim_ri[d] < 0 || static_cast<std::size_t>(dim_ri[d]) >= is->size())
                                throw std::out_of_range("expanded independent row index out of bounds");
                            const std::vector<Measurement> vals = ExpandToColumns(*is, dim_ri[d]);
                            row.fields.insert(row.fields.end(), vals.begin(), vals.end());
                        }
                        if (dep_series != nullptr)
                        {
                            if (leaf_row.row_flat < 0 || static_cast<std::size_t>(leaf_row.row_flat) >= dep_series->size())
                                throw std::out_of_range("dependent data row index out of bounds");
                            const std::vector<Measurement> vals = ExpandToColumns(*dep_series, leaf_row.row_flat);
                            row.fields.insert(row.fields.end(), vals.begin(), vals.end());
                        }
                        result.push_back(std::move(row));
                    },
                    start, end);
                return result;
            });
    }
} // namespace xdataset
