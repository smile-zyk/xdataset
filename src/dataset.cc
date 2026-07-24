#include "dataset.h"

#include <sstream>
#include <stdexcept>

namespace xdataset
{
    Dataset::Dataset(std::string name)
        : name_(std::move(name))
    {}

    std::vector<std::string> Dataset::split_path(const std::string& path)
    {
        std::vector<std::string> parts;
        if (path.empty()) return parts;
        std::istringstream stream(path);
        std::string segment;
        while (std::getline(stream, segment, '/'))
            if (!segment.empty()) parts.push_back(segment);
        return parts;
    }

    const Group* Dataset::navigate(const std::string& path) const
    {
        auto parts = split_path(path);
        const Group* node = &root_;
        for (const auto& seg : parts)
        {
            auto it = node->subgroups.find(seg);
            if (it == node->subgroups.end()) return nullptr;
            node = it->second.get();
        }
        return node;
    }

    Group* Dataset::navigate(const std::string& path)
    {
        auto parts = split_path(path);
        Group* node = &root_;
        for (const auto& seg : parts)
        {
            auto it = node->subgroups.find(seg);
            if (it == node->subgroups.end()) return nullptr;
            node = it->second.get();
        }
        return node;
    }

    Group& Dataset::navigate_or_create(const std::string& path)
    {
        auto parts = split_path(path);
        if (parts.empty())
            throw std::invalid_argument("block path must not be empty");
        Group* node = &root_;
        for (std::size_t i = 0; i + 1 < parts.size(); ++i)
        {
            auto& child = node->subgroups[parts[i]];
            if (!child) child = std::unique_ptr<Group>(new Group());
            else if (child->is_leaf())
                throw std::invalid_argument("path segment '" + parts[i] + "' is already a leaf Block");
            node = child.get();
        }
        const auto& block_name = parts.back();
        auto& child = node->subgroups[block_name];
        if (!child) child = std::unique_ptr<Group>(new Group());
        else if (child->is_leaf())
            throw std::invalid_argument("duplicate Block at path: " + path);
        return *child;
    }

    void Dataset::collect_block_paths(const Group& group, const std::string& prefix, std::vector<std::string>& paths) const
    {
        for (const auto& kv : group.subgroups)
        {
            const std::string full = prefix.empty() ? kv.first : prefix + "/" + kv.first;
            const Group& child = *kv.second;
            if (child.is_leaf()) paths.push_back(full);
            collect_block_paths(child, full, paths);
        }
    }

    std::size_t Dataset::collect_block_count(const Group& group) const
    {
        std::size_t count = 0;
        for (const auto& kv : group.subgroups)
        {
            if (kv.second->is_leaf()) ++count;
            count += collect_block_count(*kv.second);
        }
        return count;
    }

    Block& Dataset::AddBlock(const std::string& path, const BlockCreateInfo& block_info)
    {
        Group& leaf = navigate_or_create(path);
        std::string name = path;
        for (auto& ch : name)
            if (ch == '/') ch = '.';
        leaf.block = std::unique_ptr<Block>(new Block(block_info));
        leaf.block->set_name(name);
        return *leaf.block;
    }

    Block& Dataset::AddBlock(const std::string& path, BlockCreateInfo&& block_info)
    {
        return AddBlock(path, static_cast<const BlockCreateInfo&>(block_info));
    }

    std::size_t Dataset::RemoveBlock(const std::string& path)
    {
        auto parts = split_path(path);
        if (parts.empty()) return 0;
        Group* parent = &root_;
        for (std::size_t i = 0; i + 1 < parts.size(); ++i)
        {
            auto it = parent->subgroups.find(parts[i]);
            if (it == parent->subgroups.end()) return 0;
            parent = it->second.get();
        }
        const auto& block_name = parts.back();
        auto it = parent->subgroups.find(block_name);
        if (it == parent->subgroups.end()) return 0;
        if (!it->second->is_leaf()) return 0;
        parent->subgroups.erase(it);
        return 1;
    }

    std::size_t Dataset::RemoveGroup(const std::string& path)
    {
        auto parts = split_path(path);
        if (parts.empty()) { std::size_t n = block_count(); root_.subgroups.clear(); return n; }
        const Group* target = navigate(path);
        if (!target) return 0;
        std::size_t count = collect_block_count(*target);
        Group* parent = &root_;
        for (std::size_t i = 0; i + 1 < parts.size(); ++i)
        {
            auto it = parent->subgroups.find(parts[i]);
            if (it == parent->subgroups.end()) return count;
            parent = it->second.get();
        }
        parent->subgroups.erase(parts.back());
        return count;
    }

    bool Dataset::HasBlock(const std::string& path) const
    {
        const Group* g = navigate(path);
        return g != nullptr && g->is_leaf();
    }

    bool Dataset::HasGroup(const std::string& path) const { return navigate(path) != nullptr; }

    bool Dataset::HasUniqueDataArray(const std::string& data_array_name) const
    {
        std::size_t count = 0;
        std::vector<std::string> paths = GetAllBlockPaths();
        for (const auto& p : paths)
        {
            const Group* g = navigate(p);
            if (!g || !g->is_leaf()) continue;
            const Block& block = *g->block;
            for (const auto& n : block.independents())
                if (n == data_array_name && ++count > 1) return false;
            for (const auto& n : block.dependents())
                if (n == data_array_name && ++count > 1) return false;
        }
        return count == 1;
    }

    const DataArray& Dataset::GetDataArray(const std::string& block_path, const std::string& data_array_name)
    {
        Block& block = GetBlock(block_path);
        return block.GetOrCreateDataArray(data_array_name);
    }

    const DataArray& Dataset::GetDataArray(const std::string& data_array_name)
    {
        std::string path = find_unique_data_array(data_array_name);
        return GetDataArray(path, data_array_name);
    }

    const Block& Dataset::GetBlock(const std::string& path) const
    {
        const Group* g = navigate(path);
        if (!g || !g->is_leaf()) throw std::out_of_range("block not found: " + path);
        return *g->block;
    }

    Block& Dataset::GetBlock(const std::string& path)
    {
        Group* g = navigate(path);
        if (!g || !g->is_leaf()) throw std::out_of_range("block not found: " + path);
        return *g->block;
    }

    std::vector<std::string> Dataset::GetDataArrayNames(const std::string& block_path) const
    {
        const Block& block = GetBlock(block_path);
        std::vector<std::string> names;
        for (const auto& n : block.independents()) names.push_back(n);
        for (const auto& n : block.dependents()) names.push_back(n);
        return names;
    }

    std::vector<std::string> Dataset::GetBlockNames(const std::string& group_path) const
    {
        const Group* g = navigate(group_path);
        if (!g) throw std::out_of_range("group not found: " + group_path);
        std::vector<std::string> names;
        for (const auto& kv : g->subgroups)
            if (kv.second->is_leaf()) names.push_back(kv.first);
        return names;
    }

    std::vector<std::string> Dataset::GetGroupNames(const std::string& group_path) const
    {
        const Group* g = navigate(group_path);
        if (!g) throw std::out_of_range("group not found: " + group_path);
        std::vector<std::string> names;
        for (const auto& kv : g->subgroups) names.push_back(kv.first);
        return names;
    }

    std::vector<std::string> Dataset::GetAllBlockPaths() const
    {
        std::vector<std::string> paths;
        collect_block_paths(root_, "", paths);
        return paths;
    }

    std::size_t Dataset::block_count() const { return collect_block_count(root_); }

    std::string Dataset::find_unique_data_array(const std::string& data_array_name) const
    {
        std::string found_path;
        std::size_t count = 0;
        std::vector<std::string> paths = GetAllBlockPaths();
        for (const auto& p : paths)
        {
            const Group* g = navigate(p);
            if (!g || !g->is_leaf()) continue;
            const Block& block = *g->block;
            for (const auto& n : block.independents())
                if (n == data_array_name) { found_path = p; ++count; }
            for (const auto& n : block.dependents())
                if (n == data_array_name) { found_path = p; ++count; }
        }
        if (count == 0)
            throw std::invalid_argument("DataArray '" + data_array_name + "' not found in any block");
        if (count > 1)
            throw std::invalid_argument("DataArray '" + data_array_name + "' is not unique (found in " + std::to_string(count) + " places); use the full block_path/data_array_name syntax");
        return found_path;
    }
}
