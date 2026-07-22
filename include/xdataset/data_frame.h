#ifndef DATA_FRAME_H
#define DATA_FRAME_H

#include <functional>
#include <string>
#include <vector>

#include "data_series.h"
#include "measurement.h"

namespace xdataset
{
    class DataSeries;

    // =========================================================================
    // DataFrameRow -- a single row in a DataFrame table
    // =========================================================================
    struct XDATASET_API DataFrameRow
    {
        std::vector<Index>       multi_index;
        std::vector<Measurement> fields;

        std::string FormatMultiIndex() const;
    };

    // =========================================================================
    // DataFrame -- lazy-loading tabular container
    // =========================================================================
    //
    // Rows are loaded on demand in fixed-size chunks.  Derived classes
    // (BlockDataFrame, DataArrayDataFrame) provide the RowGenerator that
    // populates rows from their respective data sources.
    // =========================================================================

    class XDATASET_API DataFrame
    {
    public:
        const std::vector<std::string>& headers()   const { return headers_;   }
        std::size_t                     row_count() const { return total_rows_; }

        virtual const DataFrameRow& GetRow(Index row) const;

        std::string ToCsv() const;
        void        WriteToCsv(const std::string& file_path) const;

    protected:
        using RowGenerator = std::function<std::vector<DataFrameRow>(
            Index start_row,
            Index end_row)>;

        DataFrame() = default;

        void ConfigureDynamic(std::vector<std::string> headers,
                              std::size_t total_rows,
                              RowGenerator generator,
                              std::size_t chunk_size = 256);

        /// Configure with pre-built rows -- no lazy loading, no generator.
        /// Suitable for small DataFrames where caching is unnecessary.
        void ConfigureStatic(std::vector<std::string> headers,
                             std::vector<DataFrameRow> rows);

    private:
        void EnsureChunkLoaded(Index chunk_idx) const;

        std::vector<std::string>   headers_;
        std::size_t                total_rows_  = 0;
        std::size_t                chunk_size_  = 256;
        RowGenerator               generator_;

        mutable std::vector<DataFrameRow>  rows_;
        mutable std::vector<bool>           loaded_chunks_;
    };

    // =========================================================================
    // BlockDataFrame -- DataFrame backed by a Block's independent/dependent data
    // =========================================================================
    class Block;
    class XDATASET_API BlockDataFrame : public DataFrame
    {
    public:
        explicit BlockDataFrame(const Block& block);
    };

    // =========================================================================
    // DataArrayDataFrame -- DataFrame backed by a DataArray's data
    // =========================================================================
    class DataArray;
    class XDATASET_API DataArrayDataFrame : public DataFrame
    {
    public:
        explicit DataArrayDataFrame(const DataArray& variable);
    };

    // =========================================================================
    // MeasurementDataFrame -- DataFrame that displays a single Measurement
    // =========================================================================
    //
    // Always has exactly 1 row.  The number of columns depends on the
    // Measurement's kind and shape:
    //   - Scalar -> 1 column (the value itself)
    //   - Vector -> width columns (one per element)
    //   - Matrix -> rows x cols columns (flattened row-major)
    //
    // Unlike BlockDataFrame / DataArrayDataFrame, there is no lazy loading
    // or chunk cache — the single row is stored eagerly.
    // =========================================================================
    class XDATASET_API MeasurementDataFrame : public DataFrame
    {
    public:
        MeasurementDataFrame(const Measurement& measurement, std::string name);

        const DataFrameRow& GetRow(Index row) const override;

    private:
        DataFrameRow row_;
    };

} // namespace xdataset

#endif  // DATA_FRAME_H
