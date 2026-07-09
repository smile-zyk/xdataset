#include "block_test_helpers.h"

#include <gtest/gtest.h>

namespace xdataset
{
    using namespace test_helpers;

    TEST(VariableDataTableDataTest, IndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeValueRichCreateInfo());
        std::shared_ptr<VariableData> y_data = block.GetOrCreateVariableData("y");
        ASSERT_NE(y_data, nullptr);

        const TableData table = y_data->ToTableData();
        ASSERT_EQ(table.metadata.headers.size(), 2u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");

        ASSERT_EQ(table.rows.size(), 6u);
        ASSERT_EQ(table.metadata.multi_indices.size(), 6u);
        EXPECT_EQ(table.metadata.multi_indices[0][0], 0u);
        EXPECT_EQ(table.metadata.multi_indices[0][1], 0u);
        EXPECT_EQ(table.rows[0][0], "10");
        EXPECT_EQ(table.rows[0][1], "1");

        EXPECT_EQ(table.metadata.multi_indices[3][0], 1u);
        EXPECT_EQ(table.metadata.multi_indices[3][1], 0u);
        EXPECT_EQ(table.rows[3][0], "20");
        EXPECT_EQ(table.rows[3][1], "1");
    }

    TEST(VariableDataTableDataTest, DependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeValueRichCreateInfo());
        std::shared_ptr<VariableData> z_data = block.GetOrCreateVariableData("z");
        ASSERT_NE(z_data, nullptr);

        const TableData table = z_data->ToTableData();
        ASSERT_EQ(table.metadata.headers.size(), 3u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");
        EXPECT_EQ(table.metadata.headers[2], "data");

        ASSERT_EQ(table.rows.size(), 6u);
        ASSERT_EQ(table.metadata.multi_indices.size(), 6u);
        EXPECT_EQ(table.metadata.multi_indices[0][0], 0u);
        EXPECT_EQ(table.metadata.multi_indices[0][1], 0u);
        EXPECT_EQ(table.rows[0][0], "10");
        EXPECT_EQ(table.rows[0][1], "1");
        EXPECT_EQ(table.rows[0][2], "100");
        EXPECT_EQ(table.metadata.multi_indices[5][0], 1u);
        EXPECT_EQ(table.metadata.multi_indices[5][1], 2u);
        EXPECT_EQ(table.rows[5][2], "105");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("multi_index,x,y,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,2]\",20,3,105"), std::string::npos);
    }

    TEST(VariableDataTableDataTest, JaggedIndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<VariableData> y_data = block.GetOrCreateVariableData("y");
        ASSERT_NE(y_data, nullptr);

        const TableData table = y_data->ToTableData();
        ASSERT_EQ(table.metadata.headers.size(), 2u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");

        ASSERT_EQ(table.rows.size(), 3u);
        ASSERT_EQ(table.metadata.multi_indices.size(), 3u);

        EXPECT_EQ(table.metadata.multi_indices[0], std::vector<std::size_t>({0u, 0u}));
        EXPECT_EQ(table.rows[0], std::vector<std::string>({"10", "1"}));

        EXPECT_EQ(table.metadata.multi_indices[1], std::vector<std::size_t>({1u, 0u}));
        EXPECT_EQ(table.rows[1], std::vector<std::string>({"20", "2"}));

        EXPECT_EQ(table.metadata.multi_indices[2], std::vector<std::size_t>({1u, 1u}));
        EXPECT_EQ(table.rows[2], std::vector<std::string>({"20", "3"}));

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("multi_index,x,y"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3"), std::string::npos);
    }

    TEST(VariableDataTableDataTest, JaggedDependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<VariableData> z_data = block.GetOrCreateVariableData("z");
        ASSERT_NE(z_data, nullptr);

        const TableData table = z_data->ToTableData();
        ASSERT_EQ(table.metadata.headers.size(), 3u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");
        EXPECT_EQ(table.metadata.headers[2], "data");

        ASSERT_EQ(table.rows.size(), 3u);
        ASSERT_EQ(table.metadata.multi_indices.size(), 3u);

        EXPECT_EQ(table.metadata.multi_indices[0], std::vector<std::size_t>({0u, 0u}));
        EXPECT_EQ(table.rows[0], std::vector<std::string>({"10", "1", "100"}));

        EXPECT_EQ(table.metadata.multi_indices[1], std::vector<std::size_t>({1u, 0u}));
        EXPECT_EQ(table.rows[1], std::vector<std::string>({"20", "2", "101"}));

        EXPECT_EQ(table.metadata.multi_indices[2], std::vector<std::size_t>({1u, 1u}));
        EXPECT_EQ(table.rows[2], std::vector<std::string>({"20", "3", "102"}));

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("multi_index,x,y,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3,102"), std::string::npos);
    }

    TEST(VariableDataTableDataTest, InterleavedJaggedIndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<VariableData> z_data = block.GetOrCreateVariableData("z");
        ASSERT_NE(z_data, nullptr);

        const TableData table = z_data->ToTableData();
        ASSERT_EQ(table.metadata.headers.size(), 3u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");
        EXPECT_EQ(table.metadata.headers[2], "z");

        ASSERT_EQ(table.rows.size(), 6u);
        ASSERT_EQ(table.metadata.multi_indices.size(), 6u);

        EXPECT_EQ(table.metadata.multi_indices[0], std::vector<std::size_t>({0u, 0u, 0u}));
        EXPECT_EQ(table.rows[0], std::vector<std::string>({"10", "1", "100"}));

        EXPECT_EQ(table.metadata.multi_indices[1], std::vector<std::size_t>({0u, 0u, 1u}));
        EXPECT_EQ(table.rows[1], std::vector<std::string>({"10", "1", "200"}));

        EXPECT_EQ(table.metadata.multi_indices[2], std::vector<std::size_t>({1u, 0u, 0u}));
        EXPECT_EQ(table.rows[2], std::vector<std::string>({"20", "2", "100"}));

        EXPECT_EQ(table.metadata.multi_indices[3], std::vector<std::size_t>({1u, 0u, 1u}));
        EXPECT_EQ(table.rows[3], std::vector<std::string>({"20", "2", "200"}));

        EXPECT_EQ(table.metadata.multi_indices[4], std::vector<std::size_t>({1u, 1u, 0u}));
        EXPECT_EQ(table.rows[4], std::vector<std::string>({"20", "3", "100"}));

        EXPECT_EQ(table.metadata.multi_indices[5], std::vector<std::size_t>({1u, 1u, 1u}));
        EXPECT_EQ(table.rows[5], std::vector<std::string>({"20", "3", "200"}));

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("multi_index,x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1,1]\",20,3,200"), std::string::npos);
    }

    TEST(VariableDataTableDataTest, InterleavedJaggedDependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<VariableData> w_data = block.GetOrCreateVariableData("w");
        ASSERT_NE(w_data, nullptr);

        const TableData table = w_data->ToTableData();
        ASSERT_EQ(table.metadata.headers.size(), 4u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");
        EXPECT_EQ(table.metadata.headers[2], "z");
        EXPECT_EQ(table.metadata.headers[3], "data");

        ASSERT_EQ(table.rows.size(), 6u);
        ASSERT_EQ(table.metadata.multi_indices.size(), 6u);

        EXPECT_EQ(table.metadata.multi_indices[0], std::vector<std::size_t>({0u, 0u, 0u}));
        EXPECT_EQ(table.rows[0], std::vector<std::string>({"10", "1", "100", "1000"}));

        EXPECT_EQ(table.metadata.multi_indices[1], std::vector<std::size_t>({0u, 0u, 1u}));
        EXPECT_EQ(table.rows[1], std::vector<std::string>({"10", "1", "200", "1001"}));

        EXPECT_EQ(table.metadata.multi_indices[2], std::vector<std::size_t>({1u, 0u, 0u}));
        EXPECT_EQ(table.rows[2], std::vector<std::string>({"20", "2", "100", "1002"}));

        EXPECT_EQ(table.metadata.multi_indices[3], std::vector<std::size_t>({1u, 0u, 1u}));
        EXPECT_EQ(table.rows[3], std::vector<std::string>({"20", "2", "200", "1003"}));

        EXPECT_EQ(table.metadata.multi_indices[4], std::vector<std::size_t>({1u, 1u, 0u}));
        EXPECT_EQ(table.rows[4], std::vector<std::string>({"20", "3", "100", "1004"}));

        EXPECT_EQ(table.metadata.multi_indices[5], std::vector<std::size_t>({1u, 1u, 1u}));
        EXPECT_EQ(table.rows[5], std::vector<std::string>({"20", "3", "200", "1005"}));

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("multi_index,x,y,z,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1,1]\",20,3,200,1005"), std::string::npos);
    }
} // namespace xdataset