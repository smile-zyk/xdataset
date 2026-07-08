#ifndef BLOCK_H
#define BLOCK_H

#include <memory>
#include <string>
#include <vector>
#include <tsl/ordered_map.h>
#include "variable_data.h"
#include "variable_spec.h"

namespace xdataset
{
    struct BlockCreateInfo
    {
        std::string name;
        std::vector<VariableSpec> variable_specs;
    };

    class Block : public std::enable_shared_from_this<Block>
    {
    public:
        explicit Block(const std::string& name);
        
        explicit Block(const BlockCreateInfo& info);
        explicit Block(BlockCreateInfo&& info);

        ~Block();

        const std::string& name() const;
        
        const std::vector<std::string>& dependents() const;
        
        const std::vector<std::string>& independents() const;
        
        const VariableSpec& variable_spec(const std::string& name) const;

        std::shared_ptr<VariableData> GenerateVariableData(const std::string& variable_name);

    protected:
        void RegisterGeneratedVariableData(const std::shared_ptr<VariableData>& var_data);
        friend class VariableData; // Allow VariableData to access private members for block reference management
    private:
        std::string name_;
        std::vector<std::string> dependents_;
        std::vector<std::string> independents_;
        tsl::ordered_map<std::string, VariableSpec> variable_spec_map_;
        std::vector<std::shared_ptr<VariableData>> generated_variable_datas_;
    };
}

#endif  // BLOCK_H