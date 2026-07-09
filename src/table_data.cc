#include "table_data.h"

#include <fstream>
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

    std::string TableData::ToCsv() const
    {
        std::string csv;

        csv += "multi_index";
        for (const auto& header : metadata.headers)
        {
            csv.push_back(',');
            csv += EscapeCsvCell(header);
        }
        csv.push_back('\n');

        for (std::size_t row_index = 0; row_index < rows.size(); ++row_index)
        {
            csv += EscapeCsvCell([&]() {
                const std::vector<std::size_t>& multi_index = metadata.multi_indices[row_index];
                std::string text;
                text.push_back('[');
                for (std::size_t i = 0; i < multi_index.size(); ++i)
                {
                    if (i > 0)
                    {
                        text.push_back(',');
                    }
                    text += std::to_string(multi_index[i]);
                }
                text.push_back(']');
                return text;
            }());

            for (const auto& cell : rows[row_index])
            {
                csv.push_back(',');
                csv += EscapeCsvCell(cell);
            }
            csv.push_back('\n');
        }

        return csv;
    }

    void TableData::WriteToCsv(const std::string& file_path) const
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
} // namespace xdataset