#include "dataset.h"

#include <algorithm>
#include <stdexcept>

namespace xdataset
{
    // ========================================================================
    // Helpers
    // ========================================================================

    namespace
    {
        /// Collect DataArray names from a single Block (independents first,
        /// then dependents, insertion order).
        std::vector<std::string> data_array_names_in_block(const Block& block)
        {
            std::vector<std::string> names;
            for (const auto& n : block.independents())
                names.push_back(n);
            for (const auto& n : block.dependents())
                names.push_back(n);
            return names;
        }
    } // namespace

    // ========================================================================
    // Construction
    // ========================================================================

    Dataset::Dataset(std::string name)
        : name_(std::move(name))
    {}

    // ========================================================================
    // Mutation
    // ========================================================================

    Block& Dataset::AddBlock(const std::string& analysis_name,
                             const std::string& result_type,
                             const BlockCreateInfo& block_info)
    {
        ResultTypeMap& results = analyses_[analysis_name];

        auto result = results.emplace(result_type, Block(block_info));
        if (!result.second)
            throw std::invalid_argument(
                "duplicate result type '" + result_type +
                "' in analysis '" + analysis_name + "'");

        Block& block = results.at(result_type);
        block.set_name(name_ + "." + analysis_name + "." + result_type);
        return block;
    }

    Block& Dataset::AddBlock(const std::string& analysis_name,
                             const std::string& result_type,
                             BlockCreateInfo&& block_info)
    {
        ResultTypeMap& results = analyses_[analysis_name];

        auto result = results.emplace(result_type, Block(std::move(block_info)));
        if (!result.second)
            throw std::invalid_argument(
                "duplicate result type '" + result_type +
                "' in analysis '" + analysis_name + "'");

        Block& block = results.at(result_type);
        block.set_name(name_ + "." + analysis_name + "." + result_type);
        return block;
    }

    std::size_t Dataset::RemoveAnalysis(const std::string& analysis_name)
    {
        return analyses_.erase(analysis_name);
    }

    std::size_t Dataset::RemoveBlock(const std::string& analysis_name,
                                     const std::string& result_type)
    {
        if (!HasAnalysis(analysis_name))
            return 0;

        ResultTypeMap& results = analyses_.at(analysis_name);
        std::size_t removed = results.erase(result_type);

        if (results.empty())
            analyses_.erase(analysis_name);

        return removed;
    }

    // ========================================================================
    // Query
    // ========================================================================

    bool Dataset::HasAnalysis(const std::string& analysis_name) const
    {
        return analyses_.find(analysis_name) != analyses_.end();
    }

    bool Dataset::HasBlock(const std::string& analysis_name,
                           const std::string& result_type) const
    {
        auto ait = analyses_.find(analysis_name);
        if (ait == analyses_.end())
            return false;
        return ait->second.find(result_type) != ait->second.end();
    }

    bool Dataset::HasUniqueDataArray(const std::string& data_array_name) const
    {
        std::size_t count = 0;
        for (const auto& analysis_kv : analyses_)
        {
            const ResultTypeMap& results = analysis_kv.second;
            for (const auto& result_kv : results)
            {
                const Block& block = result_kv.second;
                for (const auto& n : block.independents())
                    if (n == data_array_name && ++count > 1)
                        return false;
                for (const auto& n : block.dependents())
                    if (n == data_array_name && ++count > 1)
                        return false;
            }
        }
        return count == 1;
    }

    // ========================================================================
    // Access -- explicit path
    // ========================================================================

    const DataArray& Dataset::GetDataArray(
        const std::string& analysis_name,
        const std::string& result_type,
        const std::string& data_array_name)
    {
        Block& block = GetBlock(analysis_name, result_type);
        return block.GetOrCreateDataArray(data_array_name);
    }

    const DataArray& Dataset::GetDataArray(const std::string& data_array_name)
    {
        std::pair<std::string, std::string> loc =
            FindUniqueDataArray(data_array_name);
        return GetDataArray(loc.first, loc.second, data_array_name);
    }

    // ========================================================================
    // Access -- blocks / analyses
    // ========================================================================

    const Dataset::ResultTypeMap& Dataset::GetAnalysis(
        const std::string& analysis_name) const
    {
        auto it = analyses_.find(analysis_name);
        if (it == analyses_.end())
            throw std::invalid_argument(
                "analysis not found: " + analysis_name);
        return it->second;
    }

    const Block& Dataset::GetBlock(const std::string& analysis_name,
                                   const std::string& result_type) const
    {
        const ResultTypeMap& results = GetAnalysis(analysis_name);
        auto rit = results.find(result_type);
        if (rit == results.end())
            throw std::invalid_argument(
                "result type '" + result_type + "' not found in analysis '" +
                analysis_name + "'");
        return rit->second;
    }

    Dataset::ResultTypeMap& Dataset::GetAnalysis(
        const std::string& analysis_name)
    {
        return analyses_.at(analysis_name);
    }

    Block& Dataset::GetBlock(const std::string& analysis_name,
                             const std::string& result_type)
    {
        return GetAnalysis(analysis_name).at(result_type);
    }

    // ========================================================================
    // Enumeration
    // ========================================================================

    std::vector<std::string> Dataset::GetAnalysisNames() const
    {
        std::vector<std::string> names;
        names.reserve(analyses_.size());
        for (const auto& kv : analyses_)
            names.push_back(kv.first);
        return names;
    }

    std::vector<std::string> Dataset::GetDataArrayNames(
        const std::string& analysis_name,
        const std::string& result_type) const
    {
        const Block& block = GetBlock(analysis_name, result_type);
        return data_array_names_in_block(block);
    }

    // ========================================================================
    // Private
    // ========================================================================

    std::pair<std::string, std::string> Dataset::FindUniqueDataArray(
        const std::string& data_array_name) const
    {
        std::string found_analysis;
        std::string found_result_type;
        std::size_t count = 0;

        for (const auto& analysis_kv : analyses_)
        {
            const std::string& analysis_name = analysis_kv.first;
            const ResultTypeMap& results = analysis_kv.second;
            for (const auto& result_kv : results)
            {
                const std::string& result_type = result_kv.first;
                const Block& block = result_kv.second;
                for (const auto& n : block.independents())
                {
                    if (n == data_array_name)
                    {
                        found_analysis     = analysis_name;
                        found_result_type  = result_type;
                        ++count;
                    }
                }
                for (const auto& n : block.dependents())
                {
                    if (n == data_array_name)
                    {
                        found_analysis     = analysis_name;
                        found_result_type  = result_type;
                        ++count;
                    }
                }
            }
        }

        if (count == 0)
            throw std::invalid_argument(
                "DataArray '" + data_array_name +
                "' not found in any block");
        if (count > 1)
            throw std::invalid_argument(
                "DataArray '" + data_array_name +
                "' is not unique (found in " + std::to_string(count) +
                " places); use the full analysis.result_type.name path");

        return {found_analysis, found_result_type};
    }

} // namespace xdataset
