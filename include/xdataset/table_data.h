#ifndef TABLE_DATA_H
#define TABLE_DATA_H

#include <string>
#include <vector>

namespace xdataset
{
    struct TableMetadata
    {
        std::vector<std::string> headers;
        std::vector<std::vector<std::size_t>> multi_indices;
    };

    struct TableData
    {
        TableMetadata metadata;
        std::vector<std::vector<std::string>> rows;

        std::string ToCsv() const;
        void WriteToCsv(const std::string& file_path) const;
    };
} // namespace xdataset

#endif // TABLE_DATA_H