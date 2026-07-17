#ifndef BLOCK_H
#define BLOCK_H

#include <memory>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

#include "data_series.h"
#include "dimension_spec.h"
#include "data_frame.h"
#include "data_array.h"

namespace xdataset
{
    struct IndependentSpec
    {
        std::string   name;
        DataSeries    data;
        DimensionSpec dimension;
    };

    struct DependentSpec
    {
        std::string name;
        DataSeries  data;
    };

    struct BlockCreateInfo
    {
        std::string                                name;
        std::vector<IndependentSpec> independent_specs;
        std::vector<DependentSpec>   dependent_specs;
    };

    class Block
    {
    public:
        explicit Block(const BlockCreateInfo& info);
        explicit Block(BlockCreateInfo&& info);

        const std::string& name() const;

        std::vector<std::string> dependents() const;

        std::vector<std::string> independents() const;

        const IndependentSpec& independent_spec(const std::string& name) const;

        const DependentSpec& dependent_spec(const std::string& name) const;

        std::shared_ptr<DataArray> GetOrCreateDataArray(const std::string& name);
        const DataFrame& GetOrCreateDataFrame() const;

    private:
        std::shared_ptr<DataArray> CreateDataArray(const IndependentSpec& info) const;

        void ensure_unique_name(const std::string& name) const;

        std::string                                           name_;
        tsl::ordered_map<std::string, IndependentSpec> independent_spec_map_;
        tsl::ordered_map<std::string, DependentSpec>   dependent_spec_map_;
        tsl::ordered_map<std::string, std::shared_ptr<DataArray>> data_array_cache_;
        mutable std::unique_ptr<DataFrame>                    data_frame_cache_;
    };
}

#endif  // BLOCK_H