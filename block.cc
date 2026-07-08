#include "block.h"
#include <stdexcept>

namespace xdataset
{
    Block::Block(const std::string& name)
        : name_(name), dependents_(), independents_(), variable_spec_map_(), generated_variable_datas_()
    {
    }

    Block::Block(const BlockCreateInfo& info)
        : name_(info.name), dependents_(), independents_(), variable_spec_map_(), generated_variable_datas_()
    {
        for (const auto& spec : info.variable_specs)
        {
            variable_spec_map_[spec.name()] = spec;
            if (spec.kind() == VariableKind::kDependent)
            {
                dependents_.push_back(spec.name());
            }
            else
            {
                independents_.push_back(spec.name());
            }
        }
    }

    Block::Block(BlockCreateInfo&& info)
        : name_(std::move(info.name)), dependents_(), independents_(), variable_spec_map_(), generated_variable_datas_()
    {
        for (auto& spec : info.variable_specs)
        {
            variable_spec_map_[spec.name()] = std::move(spec);
            const auto& spec_ref = variable_spec_map_[spec.name()];
            if (spec_ref.kind() == VariableKind::kDependent)
            {
                dependents_.push_back(spec_ref.name());
            }
            else
            {
                independents_.push_back(spec_ref.name());
            }
        }
    }

    Block::~Block()
    {
        // Clear block reference in all generated VariableData instances
        for (auto& var_data : generated_variable_datas_)
        {
            if (var_data)
            {
                var_data->ClearBlockReference();
            }
        }
    }

    const std::string& Block::name() const
    {
        return name_;
    }

    const std::vector<std::string>& Block::dependents() const
    {
        return dependents_;
    }

    const std::vector<std::string>& Block::independents() const
    {
        return independents_;
    }

    const VariableSpec& Block::variable_spec(const std::string& name) const
    {
        auto it = variable_spec_map_.find(name);
        if (it == variable_spec_map_.end())
        {
            throw std::invalid_argument("variable spec not found: " + name);
        }
        return it->second;
    }

    void Block::RegisterGeneratedVariableData(const std::shared_ptr<VariableData>& var_data)
    {
        if (var_data)
        {
            generated_variable_datas_.push_back(var_data);
        }
    }

    std::shared_ptr<VariableData> Block::GenerateVariableData(const std::string& variable_name)
    {
        // TODO: Implement variable data generation
        return nullptr;
    }
} // namespace xdataset
