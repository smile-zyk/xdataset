#ifndef DATA_ARRAY_H
#define DATA_ARRAY_H

#include <memory>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

#include "data_series.h"
#include "data_frame.h"
#include "multi_dimension_spec.h"
#include "multi_index_selector.h"

namespace xdataset
{
    enum class DataArrayKind
    {
        kDependent,
        kIndependent
    };

    struct DataArrayCreateInfo
    {
        std::string name;
        DataSeries data;
        tsl::ordered_map<std::string, DataSeries> indep_datas;
        MultiDimensionSpec multi_dimension_spec;
        DataArrayKind kind = DataArrayKind::kDependent;
    };

    class DataArray
    {
    public:
        /// Placeholder name for temporary / derived DataArrays.
        static const char* kUnnamed;

        explicit DataArray(const DataArrayCreateInfo& info);
        explicit DataArray(DataArrayCreateInfo&& info);

        // Copy: deep-copies data, resets the data_frame cache.
        DataArray(const DataArray& other);
        DataArray& operator=(const DataArray& other);

        // Move: default (all members are movable).
        DataArray(DataArray&&) = default;
        DataArray& operator=(DataArray&&) = default;

        const DataSeries& data() const
        {
            return data_;
        }

        const MultiDimensionSpec& multi_dimension_spec() const
        {
            return multi_dimension_spec_;
        }

        DataArrayKind kind() const
        {
            return kind_;
        }

        const std::string& name() const
        {
            return name_;
        }

        void set_name(const std::string& name);

        const DataFrame& GetOrCreateDataFrame() const;

        const tsl::ordered_map<std::string, DataSeries>& indep_datas() const
        {
            return indep_datas_;
        }

        /// Ordered names of independent variables (keys of indep_datas_).
        std::vector<std::string> indep_names() const;

        const DataSeries& indep_data(Index index) const;

        const DataSeries& indep_data(const std::string& name) const;

        DataArray indep(Index index = 1) const;

        DataArray indep(const std::string& name) const;

        DataArray at(const std::vector<MultiIndexSelector>& selectors) const;

        DataArray select(const std::vector<MultiIndexSelector>& selectors) const;

        // Standalone independent variable (no prior independents).
        static DataArray CreateIndependent(
            std::string name,
            DataSeries data);

        // Dependent variable from existing independent DataArray objects.
        static DataArray CreateDependent(
            std::string name,
            DataSeries data,
            const std::vector<const DataArray*>& indep_variables);

    private:
        std::string name_;
        DataSeries data_;
        tsl::ordered_map<std::string, DataSeries> indep_datas_;
        MultiDimensionSpec multi_dimension_spec_;
        DataArrayKind kind_;
        mutable std::unique_ptr<DataFrame> data_frame_cache_;
    };
} // namespace xdataset

// =========================================================================
//  DataArray arithmetic operators
// =========================================================================

namespace xdataset {

// DataArray – DataArray
DataArray operator+(const DataArray& lhs, const DataArray& rhs);
DataArray operator-(const DataArray& lhs, const DataArray& rhs);
DataArray operator*(const DataArray& lhs, const DataArray& rhs);
DataArray operator/(const DataArray& lhs, const DataArray& rhs);

// DataArray – Measurement (broadcast)
DataArray operator+(const DataArray& lhs, const Measurement& rhs);
DataArray operator-(const DataArray& lhs, const Measurement& rhs);
DataArray operator*(const DataArray& lhs, const Measurement& rhs);
DataArray operator/(const DataArray& lhs, const Measurement& rhs);

/// pow(base, exponent): exponent must be a dimensionless, non-String scalar Measurement.
DataArray pow(const DataArray& base, const Measurement& exp);

} // namespace xdataset

#endif // DATA_ARRAY_H
