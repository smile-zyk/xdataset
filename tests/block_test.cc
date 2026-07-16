#include "block_fixtures.h"

#include <gtest/gtest.h>

namespace xdataset
{
    using namespace block_fixtures;

    TEST(BlockConstructorTest, BuildsSpecsFromIndependentAndDependentRules)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableInfo x{"x", MakeScalarSeries(2), DimensionSpec::Uniform(2)};
        IndependentVariableInfo y{"y", MakeScalarSeries(3), DimensionSpec::Uniform(3)};
        DependentVariableInfo z{"z", MakeScalarSeries(6)};

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

        const IndependentVariableInfo& x_info = block.independent_variable("x");
        EXPECT_EQ(x_info.name, "x");
        EXPECT_TRUE(x_info.dimension.is_uniform());
        EXPECT_EQ(x_info.dimension.uniform_size(), 2u);

        // z is dependent — independent_variable("z") should throw
        EXPECT_THROW({ block.independent_variable("z"); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsDependentWithoutIndependent)
    {
        BlockCreateInfo info;
        info.name = "demo";

        DependentVariableInfo z{"z", MakeScalarSeries(1)};
        info.dependent_variables.push_back(z);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsDuplicateVariableNames)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableInfo x{"same", MakeScalarSeries(2), DimensionSpec::Uniform(2)};
        DependentVariableInfo z{"same", MakeScalarSeries(2)};

        info.independent_variables.push_back(x);
        info.dependent_variables.push_back(z);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsEmptyVariableName)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableInfo x{"", MakeScalarSeries(2), DimensionSpec::Uniform(2)};

        info.independent_variables.push_back(x);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsDependentDataSizeMismatchWithDerivedDims)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableInfo x{"x", MakeScalarSeries(2), DimensionSpec::Uniform(2)};
        IndependentVariableInfo y{"y", MakeScalarSeries(3), DimensionSpec::Uniform(3)};
        DependentVariableInfo z{"z", MakeScalarSeries(5)};

        info.independent_variables.push_back(x);
        info.independent_variables.push_back(y);
        info.dependent_variables.push_back(z);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockConstructorTest, RejectsIndependentDataSizeMismatchWithItsDim)
    {
        BlockCreateInfo info;
        info.name = "demo";

        IndependentVariableInfo x{"x", MakeScalarSeries(3), DimensionSpec::Uniform(2)};

        info.independent_variables.push_back(x);

        EXPECT_THROW({ Block b(info); }, std::invalid_argument);
    }

    TEST(BlockVariableCacheTest, ReturnsSamePointerForCachedVariable)
    {
        Block block(MakeBaseCreateInfo());

        const std::shared_ptr<Variable> first = block.GetOrCreateVariable("z");
        const std::shared_ptr<Variable> second = block.GetOrCreateVariable("z");

        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        EXPECT_EQ(first.get(), second.get());
    }

    TEST(BlockVariableCacheTest, BuildsIndependentVariableWithPrefixDims)
    {
        Block block(MakeBaseCreateInfo());

        const std::shared_ptr<Variable> x_data = block.GetOrCreateVariable("x");
        ASSERT_NE(x_data, nullptr);
        EXPECT_EQ(x_data->kind(), VariableKind::kIndependent);
        ASSERT_EQ(x_data->multi_dimension_spec().rank(), 1u);
        ASSERT_EQ(x_data->multi_dimension_spec().dims().size(), 1u);
        EXPECT_EQ(x_data->multi_dimension_spec().dims()[0].uniform_size(), 2);
        EXPECT_EQ(x_data->data().size(), 2u);

        const std::shared_ptr<Variable> y_data = block.GetOrCreateVariable("y");
        ASSERT_NE(y_data, nullptr);
        EXPECT_EQ(y_data->kind(), VariableKind::kIndependent);
        ASSERT_EQ(y_data->multi_dimension_spec().rank(), 2u);
        ASSERT_EQ(y_data->multi_dimension_spec().dims().size(), 2u);
        EXPECT_EQ(y_data->multi_dimension_spec().dims()[0].uniform_size(), 2);
        EXPECT_EQ(y_data->multi_dimension_spec().dims()[1].uniform_size(), 3);
        EXPECT_EQ(y_data->data().size(), 6u);   // expanded: 2*3
    }

    TEST(BlockVariableCacheTest, BuildsDependentVariableFromDescriptor)
    {
        Block block(MakeBaseCreateInfo());

        const std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        EXPECT_EQ(z_data->kind(), VariableKind::kDependent);
        ASSERT_EQ(z_data->multi_dimension_spec().rank(), 2u);
        ASSERT_EQ(z_data->multi_dimension_spec().dims().size(), 2u);
        EXPECT_EQ(z_data->multi_dimension_spec().dims()[0].uniform_size(), 2);
        EXPECT_EQ(z_data->multi_dimension_spec().dims()[1].uniform_size(), 3);
        EXPECT_EQ(z_data->data().size(), 6u);
    }

    TEST(BlockVariableCacheTest, BuildsJaggedVariableWithPrefixDims)
    {
        Block block(MakeJaggedCreateInfo());

        const std::shared_ptr<Variable> y_data = block.GetOrCreateVariable("y");
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

    TEST(BlockVariableCacheTest, BuildsInterleavedJaggedVariableWithPrefixDims)
    {
        Block block(MakeInterleavedCreateInfo());

        const std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
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
        EXPECT_EQ(z_data->data().size(), 6u);   // expanded: 2*3 [jagged 1+2=3 rows * 2]
    }

    TEST(BlockGridModelTest, AggregatesAllVariablesIntoOneTable)
    {
        Block block(MakeValueRichCreateInfo());
        const GridModel& table = block.grid_model();

        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "z");

        ASSERT_EQ(table.row_count(), 6u);
        EXPECT_EQ(table.GetRow(0).multi_index[0], 0u);
        EXPECT_EQ(table.GetRow(0).multi_index[1], 0u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "100");
        EXPECT_EQ(table.GetRow(5).multi_index[0], 1u);
        EXPECT_EQ(table.GetRow(5).multi_index[1], 2u);
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[2]), "105");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,2]\",20,3,105"), std::string::npos);
    }

    TEST(BlockGridModelTest, AggregatesJaggedVariablesIntoOneTable)
    {
        Block block(MakeJaggedCreateInfo());
        const GridModel& table = block.grid_model();

        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "z");

        ASSERT_EQ(table.row_count(), 3u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<Index>({0, 0}));
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "100");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<Index>({1, 0}));
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[2]), "101");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<Index>({1, 1}));
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[2]), "102");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3,102"), std::string::npos);
    }

    TEST(BlockGridModelTest, AggregatesInterleavedJaggedVariablesIntoOneTable)
    {
        Block block(MakeInterleavedCreateInfo());
        const GridModel& table = block.grid_model();

        ASSERT_EQ(table.headers().size(), 4u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "z");
        EXPECT_EQ(table.headers()[3], "w");

        ASSERT_EQ(table.row_count(), 6u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<Index>({0, 0, 0}));
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "100");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[3]), "1000");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<Index>({0, 0, 1}));
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[2]), "200");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[3]), "1001");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<Index>({1, 0, 0}));
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[2]), "100");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[3]), "1002");

        EXPECT_EQ(table.GetRow(3).multi_index, std::vector<Index>({1, 0, 1}));
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[2]), "200");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[3]), "1003");

        EXPECT_EQ(table.GetRow(4).multi_index, std::vector<Index>({1, 1, 0}));
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[2]), "100");
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[3]), "1004");

        EXPECT_EQ(table.GetRow(5).multi_index, std::vector<Index>({1, 1, 1}));
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[2]), "200");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[3]), "1005");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z,w"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1,1]\",20,3,200,1005"), std::string::npos);
    }

    // =========================================================================
    // New fixtures: string, multi-dep, single-dim, vector, matrix
    // =========================================================================

    TEST(BlockGridModelTest, AggregatesStringTypedVariables)
    {
        Block block(MakeStringTypedCreateInfo());
        const GridModel& table = block.grid_model();

        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "sx");
        EXPECT_EQ(table.headers()[1], "sy");
        EXPECT_EQ(table.headers()[2], "sz");

        ASSERT_EQ(table.row_count(), 4u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<Index>({0, 0}));
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "alpha");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "one");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "A");

        EXPECT_EQ(table.GetRow(3).multi_index, std::vector<Index>({1, 1}));
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[0]), "beta");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[1]), "two");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[2]), "D");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("sx,sy,sz"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",beta,two,D"), std::string::npos);
    }

    TEST(BlockGridModelTest, ThreeDimsWithMultipleDependents)
    {
        Block block(MakeThreeDimMultiDepCreateInfo());
        const GridModel& table = block.grid_model();

        ASSERT_EQ(table.headers().size(), 5u);
        EXPECT_EQ(table.headers()[0], "a");
        EXPECT_EQ(table.headers()[1], "b");
        EXPECT_EQ(table.headers()[2], "c");
        EXPECT_EQ(table.headers()[3], "p");
        EXPECT_EQ(table.headers()[4], "q");

        EXPECT_EQ(table.row_count(), 24u);
    }

    TEST(BlockGridModelTest, SingleIndependentBehavesCorrectly)
    {
        Block block(MakeSingleIndependentCreateInfo());
        const GridModel& table = block.grid_model();

        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "z");

        ASSERT_EQ(table.row_count(), 3u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "100");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[0]), "30");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[1]), "300");
    }

    TEST(BlockGridModelTest, VectorCellColumnsAreExpanded)
    {
        Block block(MakeVectorCellCreateInfo());
        const GridModel& table = block.grid_model();

        ASSERT_EQ(table.headers().size(), 5u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "vecs(1)");
        EXPECT_EQ(table.headers()[3], "vecs(2)");
        EXPECT_EQ(table.headers()[4], "vecs(3)");

        ASSERT_EQ(table.row_count(), 4u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[3]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[4]), "3");
    }

    TEST(BlockGridModelTest, MatrixCellColumnsAreExpanded)
    {
        Block block(MakeMatrixCellCreateInfo());
        const GridModel& table = block.grid_model();

        ASSERT_EQ(table.headers().size(), 6u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "mats(1,1)");
        EXPECT_EQ(table.headers()[3], "mats(1,2)");
        EXPECT_EQ(table.headers()[4], "mats(2,1)");
        EXPECT_EQ(table.headers()[5], "mats(2,2)");

        ASSERT_EQ(table.row_count(), 4u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[3]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[4]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[5]), "4");
    }

    TEST(BlockVariableCacheTest, MultiDependentVariablesAreBuiltCorrectly)
    {
        Block block(MakeThreeDimMultiDepCreateInfo());
        const std::shared_ptr<Variable> a = block.GetOrCreateVariable("a");
        const std::shared_ptr<Variable> p = block.GetOrCreateVariable("p");
        const std::shared_ptr<Variable> q = block.GetOrCreateVariable("q");

        EXPECT_EQ(a->kind(), VariableKind::kIndependent);
        EXPECT_EQ(a->multi_dimension_spec().rank(), 1u);
        EXPECT_EQ(p->kind(), VariableKind::kDependent);
        EXPECT_EQ(p->multi_dimension_spec().rank(), 3u);
        EXPECT_EQ(q->kind(), VariableKind::kDependent);
        EXPECT_EQ(q->multi_dimension_spec().rank(), 3u);
    }
} // namespace xdataset