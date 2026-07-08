#include "variable_data.h"
#include "block.h"

namespace xdataset
{
    VariableData::VariableData(const CellSeries& data)
        : data_(data), multi_dimension_spec_(MultiDimensionSpec().add_uniform(data_.size())), kind_(VariableKind::kDependent) {}

    VariableData::VariableData(CellSeries&& data)
        : data_(std::move(data)), multi_dimension_spec_(MultiDimensionSpec().add_uniform(data_.size())), kind_(VariableKind::kDependent) {}

    VariableData::~VariableData()
    {
        ClearBlockReference();
    }

    std::shared_ptr<VariableData> VariableData::Clone() const
    {
        auto cloned = std::make_shared<VariableData>(data_);
        cloned->multi_dimension_spec_ = multi_dimension_spec_;
        cloned->kind_ = kind_;
        cloned->indep_datas_ = indep_datas_;

        // Inherit block reference if valid
        auto block_ptr = block_.lock();
        if (block_ptr)
        {
            cloned->block_ = block_;
            block_ptr->RegisterGeneratedVariableData(cloned);
        }

        return cloned;
    }

    void VariableData::SetBlockReference(const std::shared_ptr<Block>& block)
    {
        block_ = block;
    }

    void VariableData::ClearBlockReference()
    {
        block_.reset();
    }
} // namespace xdataset
