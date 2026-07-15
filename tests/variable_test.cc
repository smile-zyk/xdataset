#include "block_fixtures.h"

#include <gtest/gtest.h>
#include <complex>

namespace xdataset
{
    using namespace block_fixtures;

    TEST(VariableGridModelTest, IndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeValueRichCreateInfo());
        std::shared_ptr<Variable> y_data = block.GetOrCreateVariable("y");
        ASSERT_NE(y_data, nullptr);

        const GridModel& table = y_data->grid_model();
        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");

        ASSERT_EQ(table.row_count(), 6u);
        EXPECT_EQ(table.GetRow(0).multi_index[0], 0u);
        EXPECT_EQ(table.GetRow(0).multi_index[1], 0u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");

        EXPECT_EQ(table.GetRow(3).multi_index[0], 1u);
        EXPECT_EQ(table.GetRow(3).multi_index[1], 0u);
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[1]), "1");
    }

    TEST(VariableGridModelTest, DependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeValueRichCreateInfo());
        std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        const GridModel& table = z_data->grid_model();
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "data");

        ASSERT_EQ(table.row_count(), 6u);
        EXPECT_EQ(table.GetRow(0).multi_index[0], 0u);
        EXPECT_EQ(table.GetRow(0).multi_index[1], 0u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "100");
        EXPECT_EQ(table.GetRow(5).multi_index[0], 1u);
        EXPECT_EQ(table.GetRow(5).multi_index[1], 2u);
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[2]), "105");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,2]\",20,3,105"), std::string::npos);
    }

    TEST(VariableGridModelTest, JaggedIndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<Variable> y_data = block.GetOrCreateVariable("y");
        ASSERT_NE(y_data, nullptr);

        const GridModel& table = y_data->grid_model();
        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");

        ASSERT_EQ(table.row_count(), 3u);
        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<std::size_t>({0u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<std::size_t>({1u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "2");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<std::size_t>({1u, 1u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[1]), "3");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3"), std::string::npos);
    }

    TEST(VariableGridModelTest, JaggedDependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        const GridModel& table = z_data->grid_model();
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "data");

        ASSERT_EQ(table.row_count(), 3u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<std::size_t>({0u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "100");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<std::size_t>({1u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[2]), "101");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<std::size_t>({1u, 1u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[2]), "102");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3,102"), std::string::npos);
    }

    TEST(VariableGridModelTest, InterleavedJaggedIndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        const GridModel& table = z_data->grid_model();
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "z");

        ASSERT_EQ(table.row_count(), 6u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<std::size_t>({0u, 0u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "100");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<std::size_t>({0u, 0u, 1u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[2]), "200");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<std::size_t>({1u, 0u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[2]), "100");

        EXPECT_EQ(table.GetRow(3).multi_index, std::vector<std::size_t>({1u, 0u, 1u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[2]), "200");

        EXPECT_EQ(table.GetRow(4).multi_index, std::vector<std::size_t>({1u, 1u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[2]), "100");

        EXPECT_EQ(table.GetRow(5).multi_index, std::vector<std::size_t>({1u, 1u, 1u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[2]), "200");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1,1]\",20,3,200"), std::string::npos);
    }

    TEST(VariableGridModelTest, InterleavedJaggedDependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> w_data = block.GetOrCreateVariable("w");
        ASSERT_NE(w_data, nullptr);

        const GridModel& table = w_data->grid_model();
        ASSERT_EQ(table.headers().size(), 4u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "z");
        EXPECT_EQ(table.headers()[3], "data");

        ASSERT_EQ(table.row_count(), 6u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<std::size_t>({0u, 0u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "100");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[3]), "1000");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<std::size_t>({0u, 0u, 1u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[2]), "200");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[3]), "1001");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<std::size_t>({1u, 0u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[2]), "100");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[3]), "1002");

        EXPECT_EQ(table.GetRow(3).multi_index, std::vector<std::size_t>({1u, 0u, 1u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[2]), "200");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[3]), "1003");

        EXPECT_EQ(table.GetRow(4).multi_index, std::vector<std::size_t>({1u, 1u, 0u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[2]), "100");
        EXPECT_EQ(GridFieldToString(table.GetRow(4).fields[3]), "1004");

        EXPECT_EQ(table.GetRow(5).multi_index, std::vector<std::size_t>({1u, 1u, 1u}));
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[2]), "200");
        EXPECT_EQ(GridFieldToString(table.GetRow(5).fields[3]), "1005");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1,1]\",20,3,200,1005"), std::string::npos);
    }

    TEST(VariableIndepTest, DependentIndepFromInsideOutByIndexAndName)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> w_data = block.GetOrCreateVariable("w");
        ASSERT_NE(w_data, nullptr);

        std::shared_ptr<Variable> indep1 = w_data->indep(1);
        ASSERT_NE(indep1, nullptr);
        EXPECT_EQ(indep1->name(), "z");
        EXPECT_EQ(indep1->kind(), VariableKind::kIndependent);
        EXPECT_EQ(indep1->indep("x")->name(), "x");
        EXPECT_EQ(indep1->indep("y")->name(), "y");
        EXPECT_EQ(indep1->multi_dimension_spec().rank(), 3u);

        std::shared_ptr<Variable> indep2 = w_data->indep(2);
        ASSERT_NE(indep2, nullptr);
        EXPECT_EQ(indep2->name(), "y");
        EXPECT_EQ(indep2->multi_dimension_spec().rank(), 2u);

        std::shared_ptr<Variable> by_name = w_data->indep("z");
        ASSERT_NE(by_name, nullptr);
        EXPECT_EQ(by_name->name(), indep1->name());
        EXPECT_EQ(by_name->multi_dimension_spec().rank(), indep1->multi_dimension_spec().rank());
    }

    TEST(VariableIndepTest, IndependentIndepOneReturnsIndexSeries)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        std::shared_ptr<Variable> self_indep = z_data->indep(1);
        ASSERT_NE(self_indep, nullptr);
        EXPECT_EQ(self_indep->name(), "z");
        EXPECT_EQ(self_indep->kind(), VariableKind::kIndependent);
        ASSERT_EQ(self_indep->data().size(), z_data->data().size());
        for (std::size_t i = 0; i < self_indep->data().size(); ++i)
        {
            EXPECT_EQ(self_indep->data().scalar_at<int>(i), static_cast<int>(i));
        }

        std::shared_ptr<Variable> from_name = z_data->indep("z");
        ASSERT_NE(from_name, nullptr);
        ASSERT_EQ(from_name->data().size(), self_indep->data().size());
        for (std::size_t i = 0; i < from_name->data().size(); ++i)
        {
            EXPECT_EQ(from_name->data().scalar_at<int>(i), self_indep->data().scalar_at<int>(i));
        }

        std::shared_ptr<Variable> indep2 = z_data->indep(2);
        ASSERT_NE(indep2, nullptr);
        EXPECT_EQ(indep2->name(), "y");
        EXPECT_EQ(indep2->multi_dimension_spec().rank(), 2u);
    }

    TEST(VariableSelectTest, DependentSelectReturnsCompleteVariable)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> w_data = block.GetOrCreateVariable("w");
        ASSERT_NE(w_data, nullptr);

        std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Equal(1));
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::In(std::vector<Index>{0, 1}));

        std::shared_ptr<Variable> selected = w_data->select(selectors);
        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(selected->kind(), VariableKind::kDependent);
        EXPECT_EQ(selected->multi_dimension_spec().rank(), 2u);

        EXPECT_EQ(selected->multi_dimension_spec().dims()[0].as_uniform()->size, 2u);
        EXPECT_EQ(selected->multi_dimension_spec().dims()[1].as_uniform()->size, 2u);

        const GridModel& table = selected->grid_model();
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "y");
        EXPECT_EQ(table.headers()[1], "z");
        EXPECT_EQ(table.headers()[2], "data");

        ASSERT_EQ(table.row_count(), 4u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "100");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "1002");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[0]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[1]), "200");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[2]), "1005");
    }

    TEST(VariableSelectTest, DependentSelectRejectsOutOfRangeIndices)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> w_data = block.GetOrCreateVariable("w");
        ASSERT_NE(w_data, nullptr);

        EXPECT_THROW(
            {
                w_data->select({MultiIndexSelector::In(std::vector<Index>{0, 2})});
            },
            std::out_of_range);
    }

    TEST(VariableSelectTest, DependentSelectRejectsOutOfRangeIndicesOnRaggedDimension)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> w_data = block.GetOrCreateVariable("w");
        ASSERT_NE(w_data, nullptr);

        EXPECT_THROW(
            {
                w_data->select(
                    {MultiIndexSelector::Any(),
                     MultiIndexSelector::In(std::vector<Index>{2}),
                     MultiIndexSelector::Any()});
            },
            std::out_of_range);
    }

    TEST(VariableSelectTest, DependentSelectKeepsSparseIndependentRows)
    {
        BlockCreateInfo info;
        info.name = "sparse-select";
        info.independent_variables.push_back(
            IndependentVariableCreateInfo{
                "x",
                MakeScalarSeriesFrom({10.0, 20.0, 30.0, 40.0}),
                DimensionSpec::Uniform(4)});
        info.dependent_variables.push_back(
            DependentVariableCreateInfo{
                "z",
                MakeScalarSeriesFrom({100.0, 200.0, 300.0, 400.0})});

        Block block(info);
        std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::In(std::vector<Index>{1, 3}));

        std::shared_ptr<Variable> selected = z_data->select(selectors);
        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(selected->kind(), VariableKind::kDependent);
        EXPECT_EQ(selected->multi_dimension_spec().rank(), 1u);
        EXPECT_EQ(selected->multi_dimension_spec().dims()[0].as_uniform()->size, 2u);

        const GridModel& table = selected->grid_model();
        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "data");

        ASSERT_EQ(table.row_count(), 2u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "200");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "40");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "400");
    }

    TEST(VariableSelectTest, DependentSelectProducesJaggedResultWhenInnerDimCollapsed)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> w_data = block.GetOrCreateVariable("w");
        ASSERT_NE(w_data, nullptr);

        // Collapse z (Uniform(2)) with Equal(0); retain x (Uniform(2)) and y (Jagged({1,2})).
        std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Equal(0));

        std::shared_ptr<Variable> selected = w_data->select(selectors);
        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(selected->kind(), VariableKind::kDependent);
        EXPECT_EQ(selected->multi_dimension_spec().rank(), 2u);

        EXPECT_NE(selected->multi_dimension_spec().dims()[0].as_uniform(), nullptr);
        EXPECT_EQ(selected->multi_dimension_spec().dims()[0].as_uniform()->size, 2u);

        EXPECT_NE(selected->multi_dimension_spec().dims()[1].as_jagged(), nullptr);
        const auto* jagged = selected->multi_dimension_spec().dims()[1].as_jagged();
        ASSERT_EQ(jagged->sizes.size(), 2u);
        EXPECT_EQ(jagged->sizes[0], 1);
        EXPECT_EQ(jagged->sizes[1], 2);

        const GridModel& table = selected->grid_model();
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "data");

        ASSERT_EQ(table.row_count(), 3u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "1000");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[2]), "1002");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[1]), "3");
        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[2]), "1004");
    }

    TEST(VariableSelectTest, IndependentSelectRejectsOutOfRangeEqualOnSelfDimension)
    {
        // For ragged independent variables, equal selection must still be in range.

        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<Variable> y_data = block.GetOrCreateVariable("y");
        ASSERT_NE(y_data, nullptr);

        std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Equal(1));

        EXPECT_THROW(
            {
                y_data->select(selectors);
            },
            std::out_of_range);

        selectors.clear();
        selectors.push_back(MultiIndexSelector::Equal(0));
        selectors.push_back(MultiIndexSelector::Equal(1));

        EXPECT_THROW(
            {
                y_data->select(selectors);
            },
            std::out_of_range);
    }

    TEST(VariableAtTest, VectorAtEqualReturnsScalarAndKeepsRowsAndDims)
    {
        VariableCreateInfo info;
        info.name = "v";
        info.kind = VariableKind::kDependent;
        info.data = CellSeries::Vectors<double>(2, 3);
        info.data.vector_at<double>(0) << 1.0, 2.0, 3.0;
        info.data.vector_at<double>(1) << 4.0, 5.0, 6.0;
        info.indep_datas["x"] = CellSeries::ScalarsFrom<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec(
            std::vector<DimensionSpec>{DimensionSpec::Uniform(2)});

        Variable v(info);
        std::shared_ptr<Variable> selected =
            v.at({MultiIndexSelector::Equal(1)});

        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(selected->data().cell_kind(), CellKind::kScalar);
        EXPECT_EQ(selected->data().size(), 2u);
        EXPECT_EQ(selected->data().scalar_at<double>(0), 2.0);
        EXPECT_EQ(selected->data().scalar_at<double>(1), 5.0);
        EXPECT_EQ(selected->multi_dimension_spec().rank(), v.multi_dimension_spec().rank());

        const GridModel& table = selected->grid_model();
        ASSERT_EQ(table.row_count(), 2u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "10");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[0]), "20");
        EXPECT_EQ(GridFieldToString(table.GetRow(1).fields[1]), "5");
    }

    TEST(VariableAtTest, MatrixAtEqualAndInReturnsVectorAndKeepsRowsAndDims)
    {
        VariableCreateInfo info;
        info.name = "m";
        info.kind = VariableKind::kDependent;
        info.data = CellSeries::Matrices<int>(2, 2, 3);
        info.data.matrix_at<int>(0) << 1, 2, 3,
                                      4, 5, 6;
        info.data.matrix_at<int>(1) << 7, 8, 9,
                                      10, 11, 12;
        info.indep_datas["x"] = CellSeries::ScalarsFrom<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec(
            std::vector<DimensionSpec>{DimensionSpec::Uniform(2)});

        Variable m(info);
        std::shared_ptr<Variable> selected = m.at(
            {MultiIndexSelector::Equal(1), MultiIndexSelector::In({0, 2})});

        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(selected->data().cell_kind(), CellKind::kVector);
        ASSERT_EQ(selected->data().cell_shape().size(), 1u);
        EXPECT_EQ(selected->data().cell_shape()[0], 2);
        EXPECT_EQ(selected->data().size(), 2u);
        EXPECT_EQ(selected->data().vector_at<int>(0)(0), 4);
        EXPECT_EQ(selected->data().vector_at<int>(0)(1), 6);
        EXPECT_EQ(selected->data().vector_at<int>(1)(0), 10);
        EXPECT_EQ(selected->data().vector_at<int>(1)(1), 12);
        EXPECT_EQ(selected->multi_dimension_spec().rank(), m.multi_dimension_spec().rank());
    }

    TEST(VariableAtTest, ScalarAtIsInvalid)
    {
        VariableCreateInfo info;
        info.name = "s";
        info.kind = VariableKind::kDependent;
        info.data = CellSeries::ScalarsFrom<double>({1.0, 2.0});
        info.indep_datas["x"] = CellSeries::ScalarsFrom<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec(
            std::vector<DimensionSpec>{DimensionSpec::Uniform(2)});

        Variable s(info);
        EXPECT_THROW(
            {
                s.at({MultiIndexSelector::Any()});
            },
            std::logic_error);
    }

    // =========================================================================
    // New fixtures: string-typed, vector/matrix cells
    // =========================================================================

    TEST(VariableGridModelTest, StringTypedIndependentTable)
    {
        Block block(MakeStringTypedCreateInfo());
        std::shared_ptr<Variable> sy_var = block.GetOrCreateVariable("sy");
        const GridModel& table = sy_var->grid_model();

        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "sx");
        EXPECT_EQ(table.headers()[1], "sy");

        ASSERT_EQ(table.row_count(), 4u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[0]), "alpha");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[1]), "one");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[0]), "beta");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[1]), "two");
    }

    TEST(VariableGridModelTest, StringTypedDependentTable)
    {
        Block block(MakeStringTypedCreateInfo());
        std::shared_ptr<Variable> sz_var = block.GetOrCreateVariable("sz");
        const GridModel& table = sz_var->grid_model();

        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "sx");
        EXPECT_EQ(table.headers()[1], "sy");
        EXPECT_EQ(table.headers()[2], "data");

        EXPECT_EQ(GridFieldToString(table.GetRow(2).fields[2]), "C");
        EXPECT_EQ(GridFieldToString(table.GetRow(3).fields[2]), "D");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("sx,sy,data"), std::string::npos);
    }

    TEST(VariableGridModelTest, VectorDependentColumnsExpand)
    {
        Block block(MakeVectorCellCreateInfo());
        std::shared_ptr<Variable> vecs_var = block.GetOrCreateVariable("vecs");
        const GridModel& table = vecs_var->grid_model();

        ASSERT_EQ(table.headers().size(), 5u);
        EXPECT_EQ(table.headers()[2], "data(1)");
        EXPECT_EQ(table.headers()[3], "data(2)");
        EXPECT_EQ(table.headers()[4], "data(3)");

        ASSERT_EQ(table.row_count(), 4u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[3]), "2");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[4]), "3");
    }

    TEST(VariableGridModelTest, MatrixDependentColumnsExpand)
    {
        Block block(MakeMatrixCellCreateInfo());
        std::shared_ptr<Variable> mats_var = block.GetOrCreateVariable("mats");
        const GridModel& table = mats_var->grid_model();

        ASSERT_EQ(table.headers().size(), 6u);
        EXPECT_EQ(table.headers()[2], "data(1,1)");
        EXPECT_EQ(table.headers()[3], "data(1,2)");
        EXPECT_EQ(table.headers()[4], "data(2,1)");
        EXPECT_EQ(table.headers()[5], "data(2,2)");

        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[5]), "4");
    }

    TEST(VariableGridModelTest, JaggedVectorDependentColumnsExpand)
    {
        Block block(MakeJaggedVectorCreateInfo());
        std::shared_ptr<Variable> v_var = block.GetOrCreateVariable("v");
        const GridModel& table = v_var->grid_model();

        ASSERT_EQ(table.headers().size(), 4u);
        EXPECT_EQ(table.headers()[2], "data(1)");
        EXPECT_EQ(table.headers()[3], "data(2)");

        ASSERT_EQ(table.row_count(), 3u);
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[2]), "1");
        EXPECT_EQ(GridFieldToString(table.GetRow(0).fields[3]), "2");
    }
} // namespace xdataset