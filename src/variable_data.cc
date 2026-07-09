#include "variable_data.h"

#include <complex>
#include <sstream>
#include <stdexcept>

namespace xdataset
{
    namespace
    {
        std::string FormatMultiIndex(const std::vector<std::size_t>& index)
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
            oss << value.real() << "+" << value.imag() << "i";
            return oss.str();
        }

        std::string ScalarAtToString(const CellSeries& series, std::size_t row)
        {
            if (series.cell_kind() != CellKind::kScalar)
            {
                throw std::logic_error("table view currently requires scalar CellSeries");
            }

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

    } // namespace

    VariableData::VariableData(const VariableDataCreateInfo& info)
        : name_(info.name),
          data_(info.data),
          indep_datas_(info.indep_datas),
          multi_dimension_spec_(info.multi_dimension_spec),
          kind_(info.kind)
    {
    }

    VariableData::VariableData(VariableDataCreateInfo&& info)
        : name_(std::move(info.name)),
          data_(std::move(info.data)),
          indep_datas_(std::move(info.indep_datas)),
          multi_dimension_spec_(std::move(info.multi_dimension_spec)),
          kind_(info.kind)
    {
    }

    TableData VariableData::ToTableData() const
    {
        if (multi_dimension_spec_.empty())
        {
            throw std::logic_error("variable data table view requires non-empty dimensions");
        }

        TableData table;

        std::vector<std::pair<std::string, const CellSeries*>> indep_columns;
        indep_columns.reserve(indep_datas_.size() + 1);

        for (const auto& item : indep_datas_)
        {
            indep_columns.push_back(std::make_pair(item.first, &item.second));
            table.metadata.headers.push_back(item.first);
        }

        if (kind_ == VariableKind::kIndependent)
        {
            const std::string self_name = name_.empty() ? "self" : name_;
            indep_columns.push_back(std::make_pair(self_name, &data_));
            table.metadata.headers.push_back(self_name);
        }

        if (kind_ == VariableKind::kDependent)
        {
            table.metadata.headers.push_back("data");
        }

        const std::size_t rank = multi_dimension_spec_.rank();
        if (indep_columns.size() != rank)
        {
            throw std::logic_error("independent columns count must match MultiDimensionSpec rank");
        }

        multi_dimension_spec_.for_each_leaf_row(
            [&](const std::vector<std::size_t>& multi_index,
                const std::vector<std::size_t>& dimension_row_indices,
                std::size_t row_flat)
        {
            std::vector<std::string> row;
                row.reserve(table.metadata.headers.size());

            for (std::size_t dim = 0; dim < indep_columns.size(); ++dim)
            {
                const CellSeries* indep_series = indep_columns[dim].second;
                const std::size_t src_row = dimension_row_indices[dim];
                if (src_row >= indep_series->size())
                {
                    throw std::out_of_range("expanded independent row index out of bounds");
                }
                row.push_back(ScalarAtToString(*indep_series, src_row));
            }

            if (kind_ == VariableKind::kDependent)
            {
                if (row_flat >= data_.size())
                {
                    throw std::out_of_range("dependent data row index out of bounds");
                }
                row.push_back(ScalarAtToString(data_, row_flat));
            }

            table.metadata.multi_indices.push_back(multi_index);
            table.rows.push_back(row);
        });

        return table;
    }
} // namespace xdataset
