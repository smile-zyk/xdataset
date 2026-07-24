#include "block_fixtures.h"

#include <gtest/gtest.h>
#include <complex>

namespace xdataset
{
    using namespace block_fixtures;

    TEST(DataArrayDataFrameTest, IndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeValueRichCreateInfo());
        DataArray y_data = block.GetOrCreateDataArray("y"); 
        const DataFrame& table = y_data.GetOrCreateDataFrame("y");
        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");

        ASSERT_EQ(table.row_count(), 6u);
        EXPECT_EQ(table.GetRow(0).multi_index[0], 0);
        EXPECT_EQ(table.GetRow(0).multi_index[1], 0);
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "1");

        EXPECT_EQ(table.GetRow(3).multi_index[0], 1);
        EXPECT_EQ(table.GetRow(3).multi_index[1], 0);
        EXPECT_EQ(table.GetRow(3).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(3).fields[1].to_string(), "1");
    }

    TEST(DataArrayDataFrameTest, DependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeValueRichCreateInfo());
        DataArray z_data = block.GetOrCreateDataArray("z"); const DataFrame& table = z_data.GetOrCreateDataFrame("z");
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "z");

        ASSERT_EQ(table.row_count(), 6u);
        EXPECT_EQ(table.GetRow(0).multi_index[0], 0);
        EXPECT_EQ(table.GetRow(0).multi_index[1], 0);
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "1");
        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "100");
        EXPECT_EQ(table.GetRow(5).multi_index[0], 1);
        EXPECT_EQ(table.GetRow(5).multi_index[1], 2);
        EXPECT_EQ(table.GetRow(5).fields[2].to_string(), "105");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"1,2\",20,3,105"), std::string::npos);
    }

    TEST(DataArrayDataFrameTest, RaggedIndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeRaggedCreateInfo());
        DataArray y_data = block.GetOrCreateDataArray("y"); const DataFrame& table = y_data.GetOrCreateDataFrame("y");
        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");

        ASSERT_EQ(table.row_count(), 3u);
        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<Index>({0, 0}));
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "1");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<Index>({1, 0}));
        EXPECT_EQ(table.GetRow(1).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(1).fields[1].to_string(), "2");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<Index>({1, 1}));
        EXPECT_EQ(table.GetRow(2).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(2).fields[1].to_string(), "3");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y"), std::string::npos);
        EXPECT_NE(csv.find("\"1,1\",20,3"), std::string::npos);
    }

    TEST(DataArrayDataFrameTest, RaggedDependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeRaggedCreateInfo());
        DataArray z_data = block.GetOrCreateDataArray("z"); const DataFrame& table = z_data.GetOrCreateDataFrame();
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "data");

        ASSERT_EQ(table.row_count(), 3u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<Index>({0, 0}));
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "1");
        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "100");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<Index>({1, 0}));
        EXPECT_EQ(table.GetRow(1).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(1).fields[1].to_string(), "2");
        EXPECT_EQ(table.GetRow(1).fields[2].to_string(), "101");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<Index>({1, 1}));
        EXPECT_EQ(table.GetRow(2).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(2).fields[1].to_string(), "3");
        EXPECT_EQ(table.GetRow(2).fields[2].to_string(), "102");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,data"), std::string::npos);
        EXPECT_NE(csv.find("\"1,1\",20,3,102"), std::string::npos);
    }

    TEST(DataArrayDataFrameTest, InterleavedRaggedIndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeInterleavedCreateInfo());
        DataArray z_data = block.GetOrCreateDataArray("z"); const DataFrame& table = z_data.GetOrCreateDataFrame("z");
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "z");

        ASSERT_EQ(table.row_count(), 6u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<Index>({0, 0, 0}));
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "1");
        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "100");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<Index>({0, 0, 1}));
        EXPECT_EQ(table.GetRow(1).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(1).fields[1].to_string(), "1");
        EXPECT_EQ(table.GetRow(1).fields[2].to_string(), "200");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<Index>({1, 0, 0}));
        EXPECT_EQ(table.GetRow(2).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(2).fields[1].to_string(), "2");
        EXPECT_EQ(table.GetRow(2).fields[2].to_string(), "100");

        EXPECT_EQ(table.GetRow(3).multi_index, std::vector<Index>({1, 0, 1}));
        EXPECT_EQ(table.GetRow(3).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(3).fields[1].to_string(), "2");
        EXPECT_EQ(table.GetRow(3).fields[2].to_string(), "200");

        EXPECT_EQ(table.GetRow(4).multi_index, std::vector<Index>({1, 1, 0}));
        EXPECT_EQ(table.GetRow(4).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(4).fields[1].to_string(), "3");
        EXPECT_EQ(table.GetRow(4).fields[2].to_string(), "100");

        EXPECT_EQ(table.GetRow(5).multi_index, std::vector<Index>({1, 1, 1}));
        EXPECT_EQ(table.GetRow(5).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(5).fields[1].to_string(), "3");
        EXPECT_EQ(table.GetRow(5).fields[2].to_string(), "200");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"1,1,1\",20,3,200"), std::string::npos);
    }

    TEST(DataArrayDataFrameTest, InterleavedRaggedDependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeInterleavedCreateInfo());
        DataArray w_data = block.GetOrCreateDataArray("w"); const DataFrame& table = w_data.GetOrCreateDataFrame();
        ASSERT_EQ(table.headers().size(), 4u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "z");
        EXPECT_EQ(table.headers()[3], "data");

        ASSERT_EQ(table.row_count(), 6u);

        EXPECT_EQ(table.GetRow(0).multi_index, std::vector<Index>({0, 0, 0}));
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "1");
        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "100");
        EXPECT_EQ(table.GetRow(0).fields[3].to_string(), "1 K");

        EXPECT_EQ(table.GetRow(1).multi_index, std::vector<Index>({0, 0, 1}));
        EXPECT_EQ(table.GetRow(1).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(1).fields[1].to_string(), "1");
        EXPECT_EQ(table.GetRow(1).fields[2].to_string(), "200");
        EXPECT_EQ(table.GetRow(1).fields[3].to_string(), "1.001 K");

        EXPECT_EQ(table.GetRow(2).multi_index, std::vector<Index>({1, 0, 0}));
        EXPECT_EQ(table.GetRow(2).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(2).fields[1].to_string(), "2");
        EXPECT_EQ(table.GetRow(2).fields[2].to_string(), "100");
        EXPECT_EQ(table.GetRow(2).fields[3].to_string(), "1.002 K");

        EXPECT_EQ(table.GetRow(3).multi_index, std::vector<Index>({1, 0, 1}));
        EXPECT_EQ(table.GetRow(3).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(3).fields[1].to_string(), "2");
        EXPECT_EQ(table.GetRow(3).fields[2].to_string(), "200");
        EXPECT_EQ(table.GetRow(3).fields[3].to_string(), "1.003 K");

        EXPECT_EQ(table.GetRow(4).multi_index, std::vector<Index>({1, 1, 0}));
        EXPECT_EQ(table.GetRow(4).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(4).fields[1].to_string(), "3");
        EXPECT_EQ(table.GetRow(4).fields[2].to_string(), "100");
        EXPECT_EQ(table.GetRow(4).fields[3].to_string(), "1.004 K");

        EXPECT_EQ(table.GetRow(5).multi_index, std::vector<Index>({1, 1, 1}));
        EXPECT_EQ(table.GetRow(5).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(5).fields[1].to_string(), "3");
        EXPECT_EQ(table.GetRow(5).fields[2].to_string(), "200");
        EXPECT_EQ(table.GetRow(5).fields[3].to_string(), "1.005 K");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z,data"), std::string::npos);
        EXPECT_NE(csv.find("\"1,1,1\",20,3,200,1.005 K"), std::string::npos);
    }

    TEST(DataArrayIndepTest, DependentIndepFromInsideOutByIndexAndName)
    {
        Block block(MakeInterleavedCreateInfo());
        DataArray w_data = block.GetOrCreateDataArray("w");

        // indep(1) = innermost independent (z) expanded to full product
        DataArray indep1 = w_data.indep(1);
        EXPECT_EQ(indep1.data_kind(), DataArrayKind::kIndependent);
        EXPECT_EQ(indep1.multi_dimension_spec().rank(), 3u);

        // indep(2) = middle independent (y)
        DataArray indep2 = w_data.indep(2);
        EXPECT_EQ(indep2.multi_dimension_spec().rank(), 2u);

        // by name from dependent w — w has named indeps {"x", "y", "z"}
        DataArray by_name = w_data.indep("z");
        EXPECT_EQ(by_name.multi_dimension_spec().rank(), indep1.multi_dimension_spec().rank());
    }

    // IndependentIndepOneReturnsIndexSeries ??indep(1) on an Independent returns
    // a dimension-level index series (multi_index.back()).  For z (3rd indep, 3 dims
    // U2 x J{1,2} x U2, 6 rows), the index cycles 0,1,0,1,0,1.
    TEST(DataArrayIndepTest, IndependentIndepOneReturnsIndexSeries)
    {
        Block block(MakeInterleavedCreateInfo());
        DataArray z_data = block.GetOrCreateDataArray("z");
        DataArray self_indep = z_data.indep(1);   // self-reference (key is "")
        EXPECT_EQ(self_indep.data_kind(), DataArrayKind::kIndependent);
        EXPECT_EQ(self_indep.data().size(), 6u);   // expanded product

        // Index series: last-dim cycle 0,1 repeating
        const int expected[] = {0, 1, 0, 1, 0, 1};
        for (std::size_t i = 0; i < 6; ++i)
            EXPECT_EQ(self_indep.data().scalar_at<int>(i), expected[i]);

        // indep(1) is equivalent on independent DataArray
        DataArray alt_indep = z_data.indep(1);
        ASSERT_EQ(alt_indep.data().size(), self_indep.data().size());
        for (std::size_t i = 0; i < 6; ++i)
            EXPECT_EQ(alt_indep.data().scalar_at<int>(i), self_indep.data().scalar_at<int>(i));

        DataArray indep2 = z_data.indep(2);
        EXPECT_EQ(indep2.multi_dimension_spec().rank(), 2u);
    }

    TEST(DataArraySelectTest, DependentSelectReturnsCompleteVariable)
    {
        Block block(MakeInterleavedCreateInfo());
        DataArray w_data = block.GetOrCreateDataArray("w"); std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Equal(1));
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::In(std::vector<Index>{0, 1}));

        DataArray selected = w_data.select(selectors); EXPECT_EQ(selected.data_kind(), DataArrayKind::kDependent);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), 2u);

        EXPECT_EQ(selected.multi_dimension_spec().dims()[0].as_regular()->size, 2u);
        EXPECT_EQ(selected.multi_dimension_spec().dims()[1].as_regular()->size, 2u);

        const DataFrame& table = selected.GetOrCreateDataFrame();
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "y");
        EXPECT_EQ(table.headers()[1], "z");
        EXPECT_EQ(table.headers()[2], "data");

        ASSERT_EQ(table.row_count(), 4u);
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "2");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "100");
        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "1.002 K");
        EXPECT_EQ(table.GetRow(3).fields[0].to_string(), "3");
        EXPECT_EQ(table.GetRow(3).fields[1].to_string(), "200");
        EXPECT_EQ(table.GetRow(3).fields[2].to_string(), "1.005 K");
    }

    TEST(DataArraySelectTest, DependentSelectRejectsOutOfRangeIndices)
    {
        Block block(MakeInterleavedCreateInfo());
        DataArray w_data = block.GetOrCreateDataArray("w"); EXPECT_THROW(
            {
                w_data.select({MultiIndexSelector::In(std::vector<Index>{0, 2})});
            },
            std::out_of_range);
    }

    TEST(DataArraySelectTest, DependentSelectRejectsOutOfRangeIndicesOnRaggedDimension)
    {
        Block block(MakeInterleavedCreateInfo());
        DataArray w_data = block.GetOrCreateDataArray("w"); EXPECT_THROW(
            {
                w_data.select(
                    {MultiIndexSelector::Any(),
                     MultiIndexSelector::In(std::vector<Index>{2}),
                     MultiIndexSelector::Any()});
            },
            std::out_of_range);
    }

    TEST(DataArraySelectTest, DependentSelectKeepsSparseIndependentRows)
    {
        BlockCreateInfo info;
        info.independent_specs.push_back(
            IndependentSpec{
                "x",
                MakeScalarSeriesFrom({10.0, 20.0, 30.0, 40.0}),
                DimensionSpec::Regular(4)});
        info.dependent_specs.push_back(
            DependentSpec{
                "z",
                MakeScalarSeriesFrom({100.0, 200.0, 300.0, 400.0})});

        Block block(info);
        DataArray z_data = block.GetOrCreateDataArray("z"); std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::In(std::vector<Index>{1, 3}));

        DataArray selected = z_data.select(selectors); EXPECT_EQ(selected.data_kind(), DataArrayKind::kDependent);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), 1u);
        EXPECT_EQ(selected.multi_dimension_spec().dims()[0].as_regular()->size, 2u);

        const DataFrame& table = selected.GetOrCreateDataFrame();
        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "data");

        ASSERT_EQ(table.row_count(), 2u);
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "200");
        EXPECT_EQ(table.GetRow(1).fields[0].to_string(), "40");
        EXPECT_EQ(table.GetRow(1).fields[1].to_string(), "400");
    }

    TEST(DataArraySelectTest, DependentSelectProducesJaggedResultWhenInnerDimCollapsed)
    {
        Block block(MakeInterleavedCreateInfo());
        DataArray w_data = block.GetOrCreateDataArray("w"); // Collapse z (Regular(2)) with Equal(0); retain x (Regular(2)) and y (Ragged({1,2})).
        std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Equal(0));

        DataArray selected = w_data.select(selectors); EXPECT_EQ(selected.data_kind(), DataArrayKind::kDependent);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), 2u);

        EXPECT_NE(selected.multi_dimension_spec().dims()[0].as_regular(), nullptr);
        EXPECT_EQ(selected.multi_dimension_spec().dims()[0].as_regular()->size, 2u);

        EXPECT_NE(selected.multi_dimension_spec().dims()[1].as_ragged(), nullptr);
        const auto* ragged = selected.multi_dimension_spec().dims()[1].as_ragged();
        ASSERT_EQ(ragged->sizes.size(), 2u);
        EXPECT_EQ(ragged->sizes[0], 1);
        EXPECT_EQ(ragged->sizes[1], 2);

        const DataFrame& table = selected.GetOrCreateDataFrame();
        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "x");
        EXPECT_EQ(table.headers()[1], "y");
        EXPECT_EQ(table.headers()[2], "data");

        ASSERT_EQ(table.row_count(), 3u);
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "1");
        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "1 K");
        EXPECT_EQ(table.GetRow(1).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(1).fields[1].to_string(), "2");
        EXPECT_EQ(table.GetRow(1).fields[2].to_string(), "1.002 K");
        EXPECT_EQ(table.GetRow(2).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(2).fields[1].to_string(), "3");
        EXPECT_EQ(table.GetRow(2).fields[2].to_string(), "1.004 K");
    }

    TEST(DataArraySelectTest, IndependentSelectRetainsSelfDimensionEvenWhenEqual)
    {
        // Independent DataArray: the last dimension (self) MUST NOT be collapsed
        // by Equal selector, otherwise the result has no data.

        Block block(MakeInterleavedCreateInfo());
        DataArray z_data = block.GetOrCreateDataArray("z");
        // z has dims: x(2), y(Ragged{1,2}), z-self(2)
        // Without the self-dim fix, Equal(0) on z-self would collapse the dimension,
        // leaving rank=2. With the fix, rank stays 3.
        std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Equal(0));

        DataArray selected = z_data.select(selectors);
        EXPECT_EQ(selected.data_kind(), DataArrayKind::kIndependent);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), 3u);

        const std::size_t total = selected.multi_dimension_spec().compute_cell_count();
        // 3 leaf rows: (0,0,0), (1,0,0), (1,1,0) — z-self filtered to index 0 only
        EXPECT_EQ(total, 3u);
        EXPECT_EQ(selected.data().size(), 3u);
    }

    TEST(DataArraySelectTest, IndependentSelectRejectsOutOfRangeEqualOnSelfDimension)
    {
        // For ragged independent variables, equal selection must still be in range.

        Block block(MakeRaggedCreateInfo());
        DataArray y_data = block.GetOrCreateDataArray("y"); std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Equal(1));

        EXPECT_THROW(
            {
                y_data.select(selectors);
            },
            std::out_of_range);

        selectors.clear();
        selectors.push_back(MultiIndexSelector::Equal(0));
        selectors.push_back(MultiIndexSelector::Equal(1));

        EXPECT_THROW(
            {
                y_data.select(selectors);
            },
            std::out_of_range);
    }

    TEST(DataArrayAtTest, VectorAtEqualReturnsScalarAndKeepsRowsAndDims)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries(DataKind::kVector, DataType::kReal, {3});
        info.data.resize(2);
        info.data.vector_at<double>(0) << 1.0, 2.0, 3.0;
        info.data.vector_at<double>(1) << 4.0, 5.0, 6.0;
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray v(info);
        DataArray selected =
            v.at({MultiIndexSelector::Equal(1)}); EXPECT_EQ(selected.data().data_kind(), DataKind::kScalar);
        EXPECT_EQ(selected.data().size(), 2u);
        EXPECT_EQ(selected.data().scalar_at<double>(0), 2.0);
        EXPECT_EQ(selected.data().scalar_at<double>(1), 5.0);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), v.multi_dimension_spec().rank());

        const DataFrame& table = selected.GetOrCreateDataFrame();
        ASSERT_EQ(table.row_count(), 2u);
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "2");
        EXPECT_EQ(table.GetRow(1).fields[0].to_string(), "20");
        EXPECT_EQ(table.GetRow(1).fields[1].to_string(), "5");
    }

    TEST(DataArrayAtTest, MatrixAtEqualAndInReturnsVectorAndKeepsRowsAndDims)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries(DataKind::kMatrix, DataType::kInteger, {2, 3});
        info.data.resize(2);
        info.data.matrix_at<int>(0) << 1, 2, 3,
                                      4, 5, 6;
        info.data.matrix_at<int>(1) << 7, 8, 9,
                                      10, 11, 12;
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray m(info);
        DataArray selected = m.at(
            {MultiIndexSelector::Equal(1), MultiIndexSelector::In({0, 2})}); EXPECT_EQ(selected.data().data_kind(), DataKind::kVector);
        ASSERT_EQ(selected.data().data_shape().size(), 1u);
        EXPECT_EQ(selected.data().data_shape()[0], 2);
        EXPECT_EQ(selected.data().size(), 2u);
        EXPECT_EQ(selected.data().vector_at<int>(0)(0), 4);
        EXPECT_EQ(selected.data().vector_at<int>(0)(1), 6);
        EXPECT_EQ(selected.data().vector_at<int>(1)(0), 10);
        EXPECT_EQ(selected.data().vector_at<int>(1)(1), 12);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), m.multi_dimension_spec().rank());
    }

    TEST(DataArrayAtTest, ScalarAtIsInvalid)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray s(info);
        EXPECT_THROW(
            {
                s.at({MultiIndexSelector::Any()});
            },
            std::logic_error);
    }

    TEST(DataArrayAtTest, MatrixAtBothEqualReturnsScalar)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries(DataKind::kMatrix, DataType::kInteger, {2, 3});
        info.data.resize(2);
        info.data.matrix_at<int>(0) << 1, 2, 3,
                                      4, 5, 6;
        info.data.matrix_at<int>(1) << 7, 8, 9,
                                      10, 11, 12;
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray m(info);
        // Both selectors are Equal → selected size is 1 each → auto-reduce to scalar.
        DataArray selected = m.at({MultiIndexSelector::Equal(0), MultiIndexSelector::Equal(2)});
        EXPECT_EQ(selected.data().data_kind(), DataKind::kScalar);
        EXPECT_EQ(selected.data().size(), 2u);
        EXPECT_EQ(selected.data().scalar_at<int>(0), 3);   // row 0, col 2
        EXPECT_EQ(selected.data().scalar_at<int>(1), 9);   // row 1, col 2
        EXPECT_EQ(selected.multi_dimension_spec().rank(), m.multi_dimension_spec().rank());
    }

    TEST(DataArrayAtTest, MatrixAtBothInReturnsMatrix)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries(DataKind::kMatrix, DataType::kInteger, {2, 3});
        info.data.resize(2);
        info.data.matrix_at<int>(0) << 1, 2, 3,
                                      4, 5, 6;
        info.data.matrix_at<int>(1) << 7, 8, 9,
                                      10, 11, 12;
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray m(info);
        // Both selectors select >1 index → stays matrix.
        DataArray selected = m.at(
            {MultiIndexSelector::In({0, 1}), MultiIndexSelector::In({1, 2})});
        EXPECT_EQ(selected.data().data_kind(), DataKind::kMatrix);
        ASSERT_EQ(selected.data().data_shape().size(), 2u);
        EXPECT_EQ(selected.data().data_shape()[0], 2);
        EXPECT_EQ(selected.data().data_shape()[1], 2);
        EXPECT_EQ(selected.data().size(), 2u);
        // row 0: cols 1,2 → [[2,3],[5,6]]
        EXPECT_EQ(selected.data().matrix_at<int>(0)(0, 0), 2);
        EXPECT_EQ(selected.data().matrix_at<int>(0)(0, 1), 3);
        EXPECT_EQ(selected.data().matrix_at<int>(0)(1, 0), 5);
        EXPECT_EQ(selected.data().matrix_at<int>(0)(1, 1), 6);
        // row 1: cols 1,2 → [[8,9],[11,12]]
        EXPECT_EQ(selected.data().matrix_at<int>(1)(0, 0), 8);
        EXPECT_EQ(selected.data().matrix_at<int>(1)(0, 1), 9);
        EXPECT_EQ(selected.data().matrix_at<int>(1)(1, 0), 11);
        EXPECT_EQ(selected.data().matrix_at<int>(1)(1, 1), 12);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), m.multi_dimension_spec().rank());
    }

    TEST(DataArrayAtTest, VectorAtInReturnsVector)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries(DataKind::kVector, DataType::kReal, {3});
        info.data.resize(2);
        info.data.vector_at<double>(0) << 1.0, 2.0, 3.0;
        info.data.vector_at<double>(1) << 4.0, 5.0, 6.0;
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray v(info);
        // In selects >1 index → stays vector (no auto-reduce).
        DataArray selected = v.at({MultiIndexSelector::In({0, 2})});
        EXPECT_EQ(selected.data().data_kind(), DataKind::kVector);
        ASSERT_EQ(selected.data().data_shape().size(), 1u);
        EXPECT_EQ(selected.data().data_shape()[0], 2);
        EXPECT_EQ(selected.data().size(), 2u);
        EXPECT_EQ(selected.data().vector_at<double>(0)(0), 1.0);
        EXPECT_EQ(selected.data().vector_at<double>(0)(1), 3.0);
        EXPECT_EQ(selected.data().vector_at<double>(1)(0), 4.0);
        EXPECT_EQ(selected.data().vector_at<double>(1)(1), 6.0);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), v.multi_dimension_spec().rank());
    }

    TEST(DataArrayAtTest, MatrixAtSingleSelectorAutoPadsWithAny)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries(DataKind::kMatrix, DataType::kInteger, {2, 3});
        info.data.resize(2);
        info.data.matrix_at<int>(0) << 1, 2, 3,
                                      4, 5, 6;
        info.data.matrix_at<int>(1) << 7, 8, 9,
                                      10, 11, 12;
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray m(info);
        // Single Equal selector → padded with Any → selected_rows=[1], selected_cols=[0,1,2].
        // row_reduce → vector of size 3.
        DataArray selected = m.at({MultiIndexSelector::Equal(1)});
        EXPECT_EQ(selected.data().data_kind(), DataKind::kVector);
        ASSERT_EQ(selected.data().data_shape().size(), 1u);
        EXPECT_EQ(selected.data().data_shape()[0], 3);
        EXPECT_EQ(selected.data().size(), 2u);
        EXPECT_EQ(selected.data().vector_at<int>(0)(0), 4);
        EXPECT_EQ(selected.data().vector_at<int>(0)(1), 5);
        EXPECT_EQ(selected.data().vector_at<int>(0)(2), 6);
        EXPECT_EQ(selected.data().vector_at<int>(1)(0), 10);
        EXPECT_EQ(selected.data().vector_at<int>(1)(1), 11);
        EXPECT_EQ(selected.data().vector_at<int>(1)(2), 12);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), m.multi_dimension_spec().rank());
    }

    TEST(DataArrayAtTest, VectorAtEmptySelectorAutoPadsWithAny)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries(DataKind::kVector, DataType::kReal, {3});
        info.data.resize(2);
        info.data.vector_at<double>(0) << 1.0, 2.0, 3.0;
        info.data.vector_at<double>(1) << 4.0, 5.0, 6.0;
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray v(info);
        // Empty selectors → padded with Any → selects all indices → stays vector.
        DataArray selected = v.at({});
        EXPECT_EQ(selected.data().data_kind(), DataKind::kVector);
        ASSERT_EQ(selected.data().data_shape().size(), 1u);
        EXPECT_EQ(selected.data().data_shape()[0], 3);
        EXPECT_EQ(selected.data().size(), 2u);
        EXPECT_EQ(selected.data().vector_at<double>(0)(0), 1.0);
        EXPECT_EQ(selected.data().vector_at<double>(0)(1), 2.0);
        EXPECT_EQ(selected.data().vector_at<double>(0)(2), 3.0);
        EXPECT_EQ(selected.data().vector_at<double>(1)(0), 4.0);
        EXPECT_EQ(selected.data().vector_at<double>(1)(1), 5.0);
        EXPECT_EQ(selected.data().vector_at<double>(1)(2), 6.0);
        EXPECT_EQ(selected.multi_dimension_spec().rank(), v.multi_dimension_spec().rank());
    }

    TEST(DataArrayAtTest, MatrixAtTooManySelectorsThrows)
    {
        DataArrayCreateInfo info;
        info.kind = DataArrayKind::kDependent;
        info.data = DataSeries(DataKind::kMatrix, DataType::kInteger, {2, 3});
        info.data.resize(2);
        info.data.matrix_at<int>(0) << 1, 2, 3,
                                      4, 5, 6;
        info.data.matrix_at<int>(1) << 7, 8, 9,
                                      10, 11, 12;
        info.indep_datas["x"] = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
        info.multi_dimension_spec = MultiDimensionSpec().add_regular(2);

        DataArray m(info);
        EXPECT_THROW(
            {
                m.at({MultiIndexSelector::Any(), MultiIndexSelector::Any(), MultiIndexSelector::Any()});
            },
            std::invalid_argument);
    }

    // =========================================================================
    // New fixtures: string-typed, vector/matrix cells
    // =========================================================================

    TEST(DataArrayDataFrameTest, StringTypedIndependentTable)
    {
        Block block(MakeStringTypedCreateInfo());
        DataArray sy_var = block.GetOrCreateDataArray("sy");
        const DataFrame& table = sy_var.GetOrCreateDataFrame("sy");

        ASSERT_EQ(table.headers().size(), 2u);
        EXPECT_EQ(table.headers()[0], "sx");
        EXPECT_EQ(table.headers()[1], "sy");

        ASSERT_EQ(table.row_count(), 4u);
        EXPECT_EQ(table.GetRow(0).fields[0].to_string(), "alpha");
        EXPECT_EQ(table.GetRow(0).fields[1].to_string(), "one");
        EXPECT_EQ(table.GetRow(3).fields[0].to_string(), "beta");
        EXPECT_EQ(table.GetRow(3).fields[1].to_string(), "two");
    }

    TEST(DataArrayDataFrameTest, StringTypedDependentTable)
    {
        Block block(MakeStringTypedCreateInfo());
        DataArray sz_var = block.GetOrCreateDataArray("sz");
        const DataFrame& table = sz_var.GetOrCreateDataFrame();

        ASSERT_EQ(table.headers().size(), 3u);
        EXPECT_EQ(table.headers()[0], "sx");
        EXPECT_EQ(table.headers()[1], "sy");
        EXPECT_EQ(table.headers()[2], "data");

        EXPECT_EQ(table.GetRow(2).fields[2].to_string(), "C");
        EXPECT_EQ(table.GetRow(3).fields[2].to_string(), "D");

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find("sx,sy,data"), std::string::npos);
    }

    TEST(DataArrayDataFrameTest, VectorDependentColumnsExpand)
    {
        Block block(MakeVectorCellCreateInfo());
        DataArray vecs_var = block.GetOrCreateDataArray("vecs");
        const DataFrame& table = vecs_var.GetOrCreateDataFrame();

        ASSERT_EQ(table.headers().size(), 5u);
        EXPECT_EQ(table.headers()[2], "data(1)");
        EXPECT_EQ(table.headers()[3], "data(2)");
        EXPECT_EQ(table.headers()[4], "data(3)");

        ASSERT_EQ(table.row_count(), 4u);
        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "1");
        EXPECT_EQ(table.GetRow(0).fields[3].to_string(), "2");
        EXPECT_EQ(table.GetRow(0).fields[4].to_string(), "3");
    }

    TEST(DataArrayDataFrameTest, MatrixDependentColumnsExpand)
    {
        Block block(MakeMatrixCellCreateInfo());
        DataArray mats_var = block.GetOrCreateDataArray("mats");
        const DataFrame& table = mats_var.GetOrCreateDataFrame();

        ASSERT_EQ(table.headers().size(), 6u);
        EXPECT_EQ(table.headers()[2], "data(1,1)");
        EXPECT_EQ(table.headers()[3], "data(1,2)");
        EXPECT_EQ(table.headers()[4], "data(2,1)");
        EXPECT_EQ(table.headers()[5], "data(2,2)");

        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "1");
        EXPECT_EQ(table.GetRow(0).fields[5].to_string(), "4");
    }

    TEST(DataArrayDataFrameTest, RaggedVectorDependentColumnsExpand)
    {
        Block block(MakeRaggedVectorCreateInfo());
        DataArray v_var = block.GetOrCreateDataArray("v");
        const DataFrame& table = v_var.GetOrCreateDataFrame();

        ASSERT_EQ(table.headers().size(), 4u);
        EXPECT_EQ(table.headers()[2], "data(1)");
        EXPECT_EQ(table.headers()[3], "data(2)");

        ASSERT_EQ(table.row_count(), 3u);
        EXPECT_EQ(table.GetRow(0).fields[2].to_string(), "1");
        EXPECT_EQ(table.GetRow(0).fields[3].to_string(), "2");
    }
} // namespace xdataset