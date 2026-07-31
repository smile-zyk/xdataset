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
        /// Unified storage for independent variable data and self data.
        /// The last entry (key = kSelf = "") is always the self data.
        /// For Independent DataArrays, the number of entries equals
        /// multi_dimension_spec.rank(), and all entries (including the last)
        /// are independent dimension data stored in raw (un-expanded) form.
        /// For Dependent DataArrays, the number of entries equals
        /// multi_dimension_spec.rank() + 1: the first rank entries are
        /// independent variable data (expanded), and the last (kSelf) is
        /// the dependent data (also expanded).
        tsl::ordered_map<std::string, DataSeries> datas;

        MultiDimensionSpec multi_dimension_spec;
        DataArrayKind kind = DataArrayKind::kDependent;
    };

    class XDATASET_API DataArray
    {
    public:
        /// Key used in datas_ for the self-reference entry.
        /// Always positioned last in the ordered map.
        static const char* kSelf;

        explicit DataArray(const DataArrayCreateInfo& info);
        explicit DataArray(DataArrayCreateInfo&& info);

        /// Validate a DataArrayCreateInfo: checks datas structure against
        /// multi_dimension_spec and kind.  Throws std::invalid_argument on error.
        /// The caller must canonicalize info.datas before calling.
        static void Validate(const DataArrayCreateInfo& info);

        // Copy: deep-copies data, resets the data_frame cache.
        DataArray(const DataArray& other);
        DataArray& operator=(const DataArray& other);

        // Move: default (all members are movable).
        DataArray(DataArray&&) = default;
        DataArray& operator=(DataArray&&) = default;

        /// Self data — the last entry in datas_ (key = kSelf).
        /// For Independent: raw (un-expanded) dimension data of the last dimension.
        /// For Dependent:   the dependent variable data (already expanded).
        const DataSeries& data() const
        {
            return datas_.rbegin()->second;
        }

        const MultiDimensionSpec& multi_dimension_spec() const
        {
            return multi_dimension_spec_;
        }

        DataArrayKind data_kind() const
        {
            return data_kind_;
        }

        /// Number of elements per cell: delegates to data().element_count()
        Index element_count() const
        {
            return data().element_count();
        }

        const DataFrame& GetOrCreateDataFrame(const std::string& variable_name = "data") const;

        /// Full unified data map. The last entry is always kSelf.
        const tsl::ordered_map<std::string, DataSeries>& datas() const
        {
            return datas_;
        }

        /// Independent-variable data only.
        /// - Independent: all entries (all are independent dimension data).
        /// - Dependent:   the first rank() entries (excludes kSelf).
        tsl::ordered_map<std::string, DataSeries> indep_datas() const;

        /// Ordered names of independent variables.
        std::vector<std::string> indep_names() const;

        const DataSeries& indep_data(Index index) const;

        const DataSeries& indep_data(const std::string& name) const;

        DataArray indep(Index index = 1) const;

        DataArray indep(const std::string& name) const;

        DataArray at(const std::vector<MultiIndexSelector>& selectors) const;

        DataArray select(const std::vector<MultiIndexSelector>& selectors) const;

        // Standalone independent variable (no prior independents).
        static DataArray CreateIndependent(
            DataSeries data);

        // Dependent variable with named independent DataArray objects.
        static DataArray CreateDependent(
            DataSeries data,
            const tsl::ordered_map<std::string, const DataArray*>& indep_variables);

    private:
        /// Unified data storage.  The last entry (key = kSelf) is always the
        /// self data; preceding entries are independent dimension / variable
        /// data.
        tsl::ordered_map<std::string, DataSeries> datas_;

        MultiDimensionSpec multi_dimension_spec_;
        DataArrayKind       data_kind_;
        mutable std::unique_ptr<DataFrame> data_frame_cache_;
    };
} // namespace xdataset

// =========================================================================
//  DataArray operators (delegate to OperationXxx)
// =========================================================================

namespace xdataset {

// -- arithmetic
XDATASET_API DataArray operator+(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator-(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator*(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator/(const DataArray& lhs, const DataArray& rhs);

XDATASET_API DataArray operator+(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator-(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator*(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator/(const DataArray& lhs, const Measurement& rhs);

XDATASET_API DataArray operator+(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator-(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator*(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator/(const Measurement& lhs, const DataArray& rhs);

// -- comparison
XDATASET_API DataArray operator==(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator!=(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator<(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator>(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator<=(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator>=(const DataArray& lhs, const DataArray& rhs);

XDATASET_API DataArray operator==(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator!=(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator<(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator>(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator<=(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator>=(const DataArray& lhs, const Measurement& rhs);

XDATASET_API DataArray operator==(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator!=(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator<(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator>(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator<=(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator>=(const Measurement& lhs, const DataArray& rhs);

// -- logical
XDATASET_API DataArray operator&&(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator||(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator&&(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator||(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator&&(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator||(const Measurement& lhs, const DataArray& rhs);

// -- bitwise
XDATASET_API DataArray operator&(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator|(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator^(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator&(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator|(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator^(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator&(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator|(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator^(const Measurement& lhs, const DataArray& rhs);

// -- shift
XDATASET_API DataArray operator<<(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator>>(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator<<(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator>>(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator<<(const Measurement& lhs, const DataArray& rhs);
XDATASET_API DataArray operator>>(const Measurement& lhs, const DataArray& rhs);

// -- modulo
XDATASET_API DataArray operator%(const DataArray& lhs, const DataArray& rhs);
XDATASET_API DataArray operator%(const DataArray& lhs, const Measurement& rhs);
XDATASET_API DataArray operator%(const Measurement& lhs, const DataArray& rhs);

// -- unary
XDATASET_API DataArray operator-(const DataArray& v);
XDATASET_API DataArray operator!(const DataArray& v);
XDATASET_API DataArray operator~(const DataArray& v);

/// pow(base, exponent): row-by-row, exponent must be dimensionless.
XDATASET_API DataArray pow(const DataArray& base, const DataArray& exponent);
XDATASET_API DataArray pow(const DataArray& base, const Measurement& exponent);
XDATASET_API DataArray pow(const Measurement& base, const DataArray& exponent);

} // namespace xdataset

#endif // DATA_ARRAY_H
