#ifndef TABLE_DATA_H
#define TABLE_DATA_H

#include <string>
#include <vector>

namespace xdataset
{
    class CellSeries;

    struct TableMetadata
    {
        std::vector<std::string> headers;
        std::vector<std::vector<std::size_t>> multi_indices;
    };

    struct TableData
    {
        TableMetadata metadata;
        std::vector<std::vector<std::string>> rows;

        static std::string FormatMultiIndex(const std::vector<std::size_t>& index);
        static std::string CellAtToString(const CellSeries& series, std::size_t row);
        static std::vector<std::string> ExpandHeadersForSeries(
            const std::string& base_name,
            const CellSeries& series);
        static std::vector<std::string> CellAtToColumns(const CellSeries& series, std::size_t row);

        std::string ToCsv() const;
        void WriteToCsv(const std::string& file_path) const;
    };
} // namespace xdataset

#endif // TABLE_DATA_H