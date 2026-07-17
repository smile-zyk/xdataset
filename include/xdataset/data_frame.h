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
    // DataFrameRow — a single row in a DataFrame table
    // =========================================================================
    struct DataFrameRow
    {
        std::vector<Index>       multi_index;
        std::vector<Measurement> fields;

        std::string FormatMultiIndex() const;
    };

    // =========================================================================
    // DataFrame — lazy-loading tabular container
    // =========================================================================
    //
    // Rows are loaded on demand in fixed-size chunks.  Derived classes
    // (BlockDataFrame, DataArrayDataFrame) provide the RowGenerator that
    // populates rows from their respective data sources.
    // =========================================================================

    class DataFrame
    {
    public:
        const std::vector<std::string>& headers()   const { return headers_;   }
        std::size_t                     row_count() const { return total_rows_; }

        const DataFrameRow& GetRow(Index row) const;

        std::string ToCsv() const;
        void        WriteToCsv(const std::string& file_path) const;

    protected:
        using RowGenerator = std::function<std::vector<DataFrameRow>(
            Index start_row,
            Index end_row)>;

        DataFrame() = default;

        void Configure(std::vector<std::string> headers,
                       std::size_t total_rows,
                       RowGenerator generator,
                       std::size_t chunk_size = 256);

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
    // BlockDataFrame — DataFrame backed by a Block's independent/dependent data
    // =========================================================================
    class Block;
    class BlockDataFrame : public DataFrame
    {
    public:
        explicit BlockDataFrame(const Block& block);
    };

    // =========================================================================
    // DataArrayDataFrame — DataFrame backed by a DataArray's data
    // =========================================================================
    class DataArray;
    class DataArrayDataFrame : public DataFrame
    {
    public:
        explicit DataArrayDataFrame(const DataArray& variable);
    };

} // namespace xdataset

#endif  // DATA_FRAME_H
