#include "block_test_helpers.h"

#include <gtest/gtest.h>

namespace xdataset
{
    using namespace test_helpers;

    TEST(BlockConstructorTest, BuildsSpecsFromIndependentAndDependentRules)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableCreateInfo x{"x", MakeScalarSeries(2), DimensionSpec::Uniform(2)};
        IndependentVariableCreateInfo y{"y", MakeScalarSeries(3), DimensionSpec::Uniform(3)};
        DependentVariableCreateInfo z{"z", MakeScalarSeries(6)};

        info.independent_variables.push_back(x);
        info.independent_variables.push_back(y);
        info.dependent_variables.push_back(z);

        Block block(info);

        ASSERT_EQ(block.name(), "demo");
        ASSERT_EQ(block.independents().size(), 2u);
        EXPECT_EQ(block.independents()[0], "x");
        EXPECT_EQ(block.independents()[1], "y");

        ASSERT_EQ(block.dependents().size(), 1u);
        EXPECT_EQ(block.dependents()[0], "z");

        const VariableSpec& x_spec = block.variable_spec("x");
        EXPECT_EQ(x_spec.kind(), VariableKind::kIndependent);
        EXPECT_EQ(x_spec.multi_dimension_spec().rank(), 1u);
        ASSERT_EQ(x_spec.multi_dimension_spec().dims().size(), 1u);
        EXPECT_TRUE(x_spec.multi_dimension_spec().dims()[0].is_uniform());
        EXPECT_EQ(x_spec.multi_dimension_spec().dims()[0].uniform_size(), 2);

        const VariableSpec& z_spec = block.variable_spec("z");
        EXPECT_EQ(z_spec.kind(), VariableKind::kDependent);
        EXPECT_EQ(z_spec.multi_dimension_spec().rank(), 2u);
        ASSERT_EQ(z_spec.multi_dimension_spec().dims().size(), 2u);
        EXPECT_EQ(z_spec.multi_dimension_spec().dims()[0].uniform_size(), 2);
        EXPECT_EQ(z_spec.multi_dimension_spec().dims()[1].uniform_size(), 3);
    }

    TEST(BlockConstructorTest, RejectsDependentWithoutIndependent)
    {
        BlockCreateInfo info;
        info.name = "demo";

        DependentVariableCreateInfo z{"z", MakeScalarSeries(1)};
        info.dependent_variables.push_back(z);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsDuplicateVariableNames)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableCreateInfo x{"same", MakeScalarSeries(2), DimensionSpec::Uniform(2)};
        DependentVariableCreateInfo z{"same", MakeScalarSeries(2)};

        info.independent_variables.push_back(x);
        info.dependent_variables.push_back(z);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsEmptyVariableName)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableCreateInfo x{"", MakeScalarSeries(2), DimensionSpec::Uniform(2)};

        info.independent_variables.push_back(x);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsDependentDataSizeMismatchWithDerivedDims)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableCreateInfo x{"x", MakeScalarSeries(2), DimensionSpec::Uniform(2)};
        IndependentVariableCreateInfo y{"y", MakeScalarSeries(3), DimensionSpec::Uniform(3)};
        DependentVariableCreateInfo z{"z", MakeScalarSeries(5)};

        info.independent_variables.push_back(x);
        info.independent_variables.push_back(y);
        info.dependent_variables.push_back(z);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsIndependentDataSizeMismatchWithItsDim)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableCreateInfo x{"x", MakeScalarSeries(3), DimensionSpec::Uniform(2)};

        info.independent_variables.push_back(x);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockVariableDataCacheTest, ReturnsSamePointerForCachedVariable)
    {
        Block block(MakeBaseCreateInfo());

        const std::shared_ptr<VariableData> first = block.GetOrCreateVariableData("z");
        const std::shared_ptr<VariableData> second = block.GetOrCreateVariableData("z");

        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        EXPECT_EQ(first.get(), second.get());
    }

    TEST(BlockVariableDataCacheTest, BuildsIndependentVariableDataWithPrefixDims)
    {
        Block block(MakeBaseCreateInfo());

        const std::shared_ptr<VariableData> x_data = block.GetOrCreateVariableData("x");
        ASSERT_NE(x_data, nullptr);
        EXPECT_EQ(x_data->kind(), VariableKind::kIndependent);
        ASSERT_EQ(x_data->multi_dimension_spec().rank(), 1u);
        ASSERT_EQ(x_data->multi_dimension_spec().dims().size(), 1u);
        EXPECT_EQ(x_data->multi_dimension_spec().dims()[0].uniform_size(), 2);
        EXPECT_EQ(x_data->data().size(), 2u);

        const std::shared_ptr<VariableData> y_data = block.GetOrCreateVariableData("y");
        ASSERT_NE(y_data, nullptr);
        EXPECT_EQ(y_data->kind(), VariableKind::kIndependent);
        ASSERT_EQ(y_data->multi_dimension_spec().rank(), 2u);
        ASSERT_EQ(y_data->multi_dimension_spec().dims().size(), 2u);
        EXPECT_EQ(y_data->multi_dimension_spec().dims()[0].uniform_size(), 2);
        EXPECT_EQ(y_data->multi_dimension_spec().dims()[1].uniform_size(), 3);
        EXPECT_EQ(y_data->data().size(), 3u);
    }

    TEST(BlockVariableDataCacheTest, BuildsDependentVariableDataFromSpec)
    {
        Block block(MakeBaseCreateInfo());

        const std::shared_ptr<VariableData> z_data = block.GetOrCreateVariableData("z");
        ASSERT_NE(z_data, nullptr);

        EXPECT_EQ(z_data->kind(), VariableKind::kDependent);
        ASSERT_EQ(z_data->multi_dimension_spec().rank(), 2u);
        ASSERT_EQ(z_data->multi_dimension_spec().dims().size(), 2u);
        EXPECT_EQ(z_data->multi_dimension_spec().dims()[0].uniform_size(), 2);
        EXPECT_EQ(z_data->multi_dimension_spec().dims()[1].uniform_size(), 3);
        EXPECT_EQ(z_data->data().size(), 6u);
    }

    TEST(BlockVariableDataCacheTest, BuildsJaggedVariableDataWithPrefixDims)
    {
        Block block(MakeJaggedCreateInfo());

        const std::shared_ptr<VariableData> y_data = block.GetOrCreateVariableData("y");
        ASSERT_NE(y_data, nullptr);

        EXPECT_EQ(y_data->kind(), VariableKind::kIndependent);
        ASSERT_EQ(y_data->multi_dimension_spec().rank(), 2u);
        ASSERT_EQ(y_data->multi_dimension_spec().dims().size(), 2u);
        EXPECT_TRUE(y_data->multi_dimension_spec().dims()[0].is_uniform());
        EXPECT_TRUE(y_data->multi_dimension_spec().dims()[1].is_jagged());
        EXPECT_EQ(y_data->multi_dimension_spec().dims()[0].uniform_size(), 2);
        ASSERT_EQ(y_data->multi_dimension_spec().dims()[1].jagged_sizes().size(), 2u);
        EXPECT_EQ(y_data->multi_dimension_spec().dims()[1].jagged_sizes()[0], 1);
        EXPECT_EQ(y_data->multi_dimension_spec().dims()[1].jagged_sizes()[1], 2);
        EXPECT_EQ(y_data->data().size(), 3u);
    }

    TEST(BlockVariableDataCacheTest, BuildsInterleavedJaggedVariableDataWithPrefixDims)
    {
        Block block(MakeInterleavedCreateInfo());

        const std::shared_ptr<VariableData> z_data = block.GetOrCreateVariableData("z");
        ASSERT_NE(z_data, nullptr);

        EXPECT_EQ(z_data->kind(), VariableKind::kIndependent);
        ASSERT_EQ(z_data->multi_dimension_spec().rank(), 3u);
        ASSERT_EQ(z_data->multi_dimension_spec().dims().size(), 3u);
        EXPECT_TRUE(z_data->multi_dimension_spec().dims()[0].is_uniform());
        EXPECT_TRUE(z_data->multi_dimension_spec().dims()[1].is_jagged());
        EXPECT_TRUE(z_data->multi_dimension_spec().dims()[2].is_uniform());
        EXPECT_EQ(z_data->multi_dimension_spec().dims()[0].uniform_size(), 2);
        ASSERT_EQ(z_data->multi_dimension_spec().dims()[1].jagged_sizes().size(), 2u);
        EXPECT_EQ(z_data->multi_dimension_spec().dims()[1].jagged_sizes()[0], 1);
        EXPECT_EQ(z_data->multi_dimension_spec().dims()[1].jagged_sizes()[1], 2);
        EXPECT_EQ(z_data->multi_dimension_spec().dims()[2].uniform_size(), 2);
        EXPECT_EQ(z_data->data().size(), 2u);
    }

    TEST(BlockTableDataTest, AggregatesAllVariablesIntoOneTable)
    {
        Block block(MakeValueRichCreateInfo());
        const TableData table = block.ToTableData();

        ASSERT_EQ(table.metadata.headers.size(), 3u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");
        EXPECT_EQ(table.metadata.headers[2], "z");

        ASSERT_EQ(table.rows.size(), 6u);
        ASSERT_EQ(table.metadata.multi_indices.size(), 6u);
        EXPECT_EQ(table.metadata.multi_indices[0][0], 0u);
        EXPECT_EQ(table.metadata.multi_indices[0][1], 0u);
        EXPECT_EQ(table.rows[0][0], "10");
        EXPECT_EQ(table.rows[0][1], "1");
        EXPECT_EQ(table.rows[0][2], "100");
        EXPECT_EQ(table.metadata.multi_indices[5][0], 1u);
        EXPECT_EQ(table.metadata.multi_indices[5][1], 2u);
        EXPECT_EQ(table.rows[5][0], "20");
        EXPECT_EQ(table.rows[5][1], "3");
        EXPECT_EQ(table.rows[5][2], "105");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("multi_index,x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,2]\",20,3,105"), std::string::npos);
    }

    TEST(BlockTableDataTest, AggregatesJaggedVariablesIntoOneTable)
    {
        Block block(MakeJaggedCreateInfo());
        const TableData table = block.ToTableData();

        ASSERT_EQ(table.metadata.headers.size(), 3u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");
        EXPECT_EQ(table.metadata.headers[2], "z");

        ASSERT_EQ(table.rows.size(), 3u);
        ASSERT_EQ(table.metadata.multi_indices.size(), 3u);

        EXPECT_EQ(table.metadata.multi_indices[0], std::vector<std::size_t>({0u, 0u}));
        EXPECT_EQ(table.rows[0], std::vector<std::string>({"10", "1", "100"}));

        EXPECT_EQ(table.metadata.multi_indices[1], std::vector<std::size_t>({1u, 0u}));
        EXPECT_EQ(table.rows[1], std::vector<std::string>({"20", "2", "101"}));

        EXPECT_EQ(table.metadata.multi_indices[2], std::vector<std::size_t>({1u, 1u}));
        EXPECT_EQ(table.rows[2], std::vector<std::string>({"20", "3", "102"}));

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("multi_index,x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3,102"), std::string::npos);
    }

    TEST(BlockTableDataTest, AggregatesInterleavedJaggedVariablesIntoOneTable)
    {
        Block block(MakeInterleavedCreateInfo());
        const TableData table = block.ToTableData();

        ASSERT_EQ(table.metadata.headers.size(), 4u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");
        EXPECT_EQ(table.metadata.headers[2], "z");
        EXPECT_EQ(table.metadata.headers[3], "w");

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
        EXPECT_NE(csv.find("multi_index,x,y,z,w"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1,1]\",20,3,200,1005"), std::string::npos);
    }
} // namespace xdataset