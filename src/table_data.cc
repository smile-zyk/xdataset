#include "table_data.h"
#include "cell_series.h"

#include <complex>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace xdataset
{
    namespace
    {
        template <typename T>
        std::string NumberToString(const T& value)
        {
            std::ostringstream oss;
            oss << value;
            return oss.str();
        }

        std::string ComplexToString(const std::complex<double>& value)
        {
            std::ostringstream oss;
            oss << value.real();
            if (value.imag() >= 0.0)
            {
                oss << "+";
            }
            oss << value.imag() << "i";
            return oss.str();
        }

        template <typename T>
        std::string ElementToString(const T& value)
        {
            return NumberToString(value);
        }

        template <>
        std::string ElementToString<std::complex<double>>(const std::complex<double>& value)
        {
            return ComplexToString(value);
        }

        template <>
        std::string ElementToString<std::string>(const std::string& value)
        {
            return value;
        }

        template <typename VecLike>
        std::string FormatVectorLike(const VecLike& vec, xdataset::Index width)
        {
            std::ostringstream oss;
            oss << "[";
            for (xdataset::Index i = 0; i < width; ++i)
            {
                if (i > 0)
                {
                    oss << ",";
                }
                oss << ElementToString(vec(i));
            }
            oss << "]";
            return oss.str();
        }

        template <typename MatLike>
        std::string FormatMatrixLike(const MatLike& mat, xdataset::Index rows, xdataset::Index cols)
        {
            std::ostringstream oss;
            oss << "[";
            for (xdataset::Index r = 0; r < rows; ++r)
            {
                if (r > 0)
                {
                    oss << ",";
                }
                oss << "[";
                for (xdataset::Index c = 0; c < cols; ++c)
                {
                    if (c > 0)
                    {
                        oss << ",";
                    }
                    oss << ElementToString(mat(r, c));
                }
                oss << "]";
            }
            oss << "]";
            return oss.str();
        }

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

    std::string TableData::FormatMultiIndex(const std::vector<std::size_t>& index)
    {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < index.size(); ++i)
        {
            if (i > 0)
            {
                oss << ",";
            }
            oss << index[i];
        }
        oss << "]";
        return oss.str();
    }

    std::string TableData::CellAtToString(const CellSeries& series, std::size_t row)
    {
        if (series.cell_kind() == CellKind::kScalar)
        {
            switch (series.dtype())
            {
                case DTypeTag::kReal:
                    return NumberToString(series.scalar_at<double>(row));
                case DTypeTag::kInteger:
                    return NumberToString(series.scalar_at<int>(row));
                case DTypeTag::kComplex:
                    return ComplexToString(series.scalar_at<std::complex<double>>(row));
                case DTypeTag::kString:
                    return series.scalar_at<std::string>(row);
            }
            throw std::logic_error("unsupported scalar dtype");
        }

        if (series.cell_kind() == CellKind::kVector)
        {
            const Index width = series.cell_shape()[0];
            switch (series.dtype())
            {
                case DTypeTag::kReal:
                    return FormatVectorLike(series.vector_at<double>(row), width);
                case DTypeTag::kInteger:
                    return FormatVectorLike(series.vector_at<int>(row), width);
                case DTypeTag::kComplex:
                    return FormatVectorLike(series.vector_at<std::complex<double>>(row), width);
                case DTypeTag::kString:
                    return FormatVectorLike(series.vector_at<std::string>(row), width);
            }
            throw std::logic_error("unsupported vector dtype");
        }

        if (series.cell_kind() == CellKind::kMatrix)
        {
            const Index rows = series.cell_shape()[0];
            const Index cols = series.cell_shape()[1];
            switch (series.dtype())
            {
                case DTypeTag::kReal:
                    return FormatMatrixLike(series.matrix_at<double>(row), rows, cols);
                case DTypeTag::kInteger:
                    return FormatMatrixLike(series.matrix_at<int>(row), rows, cols);
                case DTypeTag::kComplex:
                    return FormatMatrixLike(series.matrix_at<std::complex<double>>(row), rows, cols);
                case DTypeTag::kString:
                    return FormatMatrixLike(series.matrix_at<std::string>(row), rows, cols);
            }
            throw std::logic_error("unsupported matrix dtype");
        }

        throw std::logic_error("unsupported cell kind");
    }

    std::vector<std::string> TableData::ExpandHeadersForSeries(
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

    std::vector<std::string> TableData::CellAtToColumns(const CellSeries& series, std::size_t row)
    {
        std::vector<std::string> cells;
        if (series.cell_kind() == CellKind::kScalar)
        {
            cells.push_back(CellAtToString(series, row));
            return cells;
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
                    for (Index i = 0; i < width; ++i) cells.push_back(ElementToString(vec(i)));
                    return cells;
                }
                case DTypeTag::kInteger:
                {
                    const auto vec = series.vector_at<int>(row);
                    for (Index i = 0; i < width; ++i) cells.push_back(ElementToString(vec(i)));
                    return cells;
                }
                case DTypeTag::kComplex:
                {
                    const auto vec = series.vector_at<std::complex<double>>(row);
                    for (Index i = 0; i < width; ++i) cells.push_back(ElementToString(vec(i)));
                    return cells;
                }
                case DTypeTag::kString:
                {
                    const auto& vec = series.vector_at<std::string>(row);
                    for (Index i = 0; i < width; ++i) cells.push_back(ElementToString(vec(i)));
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
                    {
                        for (Index c = 0; c < cols; ++c)
                        {
                            cells.push_back(ElementToString(mat(r, c)));
                        }
                    }
                    return cells;
                }
                case DTypeTag::kInteger:
                {
                    const auto mat = series.matrix_at<int>(row);
                    for (Index r = 0; r < rows; ++r)
                    {
                        for (Index c = 0; c < cols; ++c)
                        {
                            cells.push_back(ElementToString(mat(r, c)));
                        }
                    }
                    return cells;
                }
                case DTypeTag::kComplex:
                {
                    const auto mat = series.matrix_at<std::complex<double>>(row);
                    for (Index r = 0; r < rows; ++r)
                    {
                        for (Index c = 0; c < cols; ++c)
                        {
                            cells.push_back(ElementToString(mat(r, c)));
                        }
                    }
                    return cells;
                }
                case DTypeTag::kString:
                {
                    const auto& mat = series.matrix_at<std::string>(row);
                    for (Index r = 0; r < rows; ++r)
                    {
                        for (Index c = 0; c < cols; ++c)
                        {
                            cells.push_back(ElementToString(mat(r, c)));
                        }
                    }
                    return cells;
                }
            }
            throw std::logic_error("unsupported matrix dtype");
        }

        throw std::logic_error("unsupported cell kind");
    }

    std::string TableData::ToCsv() const
    {
        std::string csv;

        for (const auto& header : metadata.headers)
        {
            csv.push_back(',');
            csv += EscapeCsvCell(header);
        }
        csv.push_back('\n');

        for (std::size_t row_index = 0; row_index < rows.size(); ++row_index)
        {
            csv += EscapeCsvCell(FormatMultiIndex(metadata.multi_indices[row_index]));

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