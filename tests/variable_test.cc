#include "block_test_helpers.h"

#include <gtest/gtest.h>
#include <complex>

namespace xdataset
{
    using namespace test_helpers;

    TEST(VariableTableDataTest, IndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeValueRichCreateInfo());
        std::shared_ptr<Variable> y_data = block.GetOrCreateVariable("y");
        ASSERT_NE(y_data, nullptr);

        const TableData& table = y_data->GetOrCreateTableData();
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

    TEST(VariableTableDataTest, DependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeValueRichCreateInfo());
        std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        const TableData& table = z_data->GetOrCreateTableData();
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
        EXPECT_NE(csv.find(",x,y,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,2]\",20,3,105"), std::string::npos);
    }

    TEST(VariableTableDataTest, JaggedIndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<Variable> y_data = block.GetOrCreateVariable("y");
        ASSERT_NE(y_data, nullptr);

        const TableData& table = y_data->GetOrCreateTableData();
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
        EXPECT_NE(csv.find(",x,y"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3"), std::string::npos);
    }

    TEST(VariableTableDataTest, JaggedDependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        const TableData& table = z_data->GetOrCreateTableData();
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
        EXPECT_NE(csv.find(",x,y,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3,102"), std::string::npos);
    }

    TEST(VariableTableDataTest, InterleavedJaggedIndependentTableExpandsPrefixDimensions)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> z_data = block.GetOrCreateVariable("z");
        ASSERT_NE(z_data, nullptr);

        const TableData& table = z_data->GetOrCreateTableData();
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
        EXPECT_NE(csv.find(",x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1,1]\",20,3,200"), std::string::npos);
    }

    TEST(VariableTableDataTest, InterleavedJaggedDependentTableContainsDataColumnAndCsv)
    {
        Block block(MakeInterleavedCreateInfo());
        std::shared_ptr<Variable> w_data = block.GetOrCreateVariable("w");
        ASSERT_NE(w_data, nullptr);

        const TableData& table = w_data->GetOrCreateTableData();
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
        EXPECT_NE(csv.find(",x,y,z,data"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1,1]\",20,3,200,1005"), std::string::npos);
    }

    TEST(TableDataStringifyTest, SupportsAllScalarDtypes)
    {
        CellSeries real_series = CellSeries::Scalars<double>(1);
        real_series.scalar_at<double>(0) = 3.5;
        EXPECT_EQ(TableData::CellAtToString(real_series, 0), "3.5");

        CellSeries int_series = CellSeries::Scalars<int>(1, 42);
        EXPECT_EQ(TableData::CellAtToString(int_series, 0), "42");

        CellSeries complex_series = CellSeries::Scalars<std::complex<double>>(1);
        complex_series.scalar_at<std::complex<double>>(0) = std::complex<double>(1.0, -2.0);
        EXPECT_EQ(TableData::CellAtToString(complex_series, 0), "1-2i");

        CellSeries string_series = CellSeries::Scalars<std::string>(1, std::string("ok"));
        EXPECT_EQ(TableData::CellAtToString(string_series, 0), "ok");
    }

    TEST(TableDataStringifyTest, SupportsAllVectorDtypes)
    {
        CellSeries real_vec = CellSeries::Vectors<double>(1, 3);
        real_vec.vector_at<double>(0) << 1.0, 2.5, -3.0;
        EXPECT_EQ(TableData::CellAtToString(real_vec, 0), "[1,2.5,-3]");

        CellSeries int_vec = CellSeries::Vectors<int>(1, 3);
        int_vec.vector_at<int>(0) << 1, 2, 3;
        EXPECT_EQ(TableData::CellAtToString(int_vec, 0), "[1,2,3]");

        CellSeries complex_vec = CellSeries::Vectors<std::complex<double>>(1, 2);
        complex_vec.vector_at<std::complex<double>>(0)(0) = std::complex<double>(1.0, 2.0);
        complex_vec.vector_at<std::complex<double>>(0)(1) = std::complex<double>(-3.0, -4.0);
        EXPECT_EQ(TableData::CellAtToString(complex_vec, 0), "[1+2i,-3-4i]");

        CellSeries string_vec = CellSeries::Vectors<std::string>(1, 2);
        string_vec.vector_at<std::string>(0)(0) = "a";
        string_vec.vector_at<std::string>(0)(1) = "b";
        EXPECT_EQ(TableData::CellAtToString(string_vec, 0), "[a,b]");
    }

    TEST(TableDataStringifyTest, SupportsAllMatrixDtypes)
    {
        CellSeries real_mat = CellSeries::Matrices<double>(1, 2, 2);
        real_mat.matrix_at<double>(0) << 1.0, 2.0,
                                        3.5, 4.0;
        EXPECT_EQ(TableData::CellAtToString(real_mat, 0), "[[1,2],[3.5,4]]");

        CellSeries int_mat = CellSeries::Matrices<int>(1, 2, 2);
        int_mat.matrix_at<int>(0) << 1, 2,
                                     3, 4;
        EXPECT_EQ(TableData::CellAtToString(int_mat, 0), "[[1,2],[3,4]]");

        CellSeries complex_mat = CellSeries::Matrices<std::complex<double>>(1, 2, 2);
        complex_mat.matrix_at<std::complex<double>>(0)(0, 0) = std::complex<double>(1.0, 1.0);
        complex_mat.matrix_at<std::complex<double>>(0)(0, 1) = std::complex<double>(2.0, -1.0);
        complex_mat.matrix_at<std::complex<double>>(0)(1, 0) = std::complex<double>(-3.0, 0.0);
        complex_mat.matrix_at<std::complex<double>>(0)(1, 1) = std::complex<double>(4.0, 2.0);
        EXPECT_EQ(TableData::CellAtToString(complex_mat, 0), "[[1+1i,2-1i],[-3+0i,4+2i]]");

        CellSeries string_mat = CellSeries::Matrices<std::string>(1, 2, 2);
        string_mat.matrix_at<std::string>(0)(0, 0) = "x";
        string_mat.matrix_at<std::string>(0)(0, 1) = "y";
        string_mat.matrix_at<std::string>(0)(1, 0) = "z";
        string_mat.matrix_at<std::string>(0)(1, 1) = "w";
        EXPECT_EQ(TableData::CellAtToString(string_mat, 0), "[[x,y],[z,w]]");
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

        const TableData& table = selected->GetOrCreateTableData();
        ASSERT_EQ(table.metadata.headers.size(), 3u);
        EXPECT_EQ(table.metadata.headers[0], "y");
        EXPECT_EQ(table.metadata.headers[1], "z");
        EXPECT_EQ(table.metadata.headers[2], "data");

        ASSERT_EQ(table.rows.size(), 4u);
        EXPECT_EQ(table.rows[0], std::vector<std::string>({"1", "100", "1002"}));
        EXPECT_EQ(table.rows[3], std::vector<std::string>({"2", "200", "1005"}));
    }

    TEST(VariableSelectTest, IndependentSelectSupportsEqualOnSelfDimension)
    {
        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<Variable> y_data = block.GetOrCreateVariable("y");
        ASSERT_NE(y_data, nullptr);

        std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Equal(1));

        std::shared_ptr<Variable> selected = y_data->select(selectors);
        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(selected->kind(), VariableKind::kIndependent);

        const TableData& table = selected->GetOrCreateTableData();
        ASSERT_EQ(table.metadata.headers.size(), 2u);
        EXPECT_EQ(table.metadata.headers[0], "x");
        EXPECT_EQ(table.metadata.headers[1], "y");

        ASSERT_EQ(table.rows.size(), 1u);
        EXPECT_EQ(table.rows[0], std::vector<std::string>({"20", "3"}));
    }
} // namespace xdataset