#include "block_fixtures.h"

#include <gtest/gtest.h>

namespace xdataset
{
    using namespace block_fixtures;

    // Test wrapper that exposes protected Configure / constructor of DataFrame.
    class TestDataFrame : public DataFrame
    {
    public:
        TestDataFrame() = default;

        void ConfigureDynamic(std::vector<std::string> h, std::size_t n,
                              RowGenerator g, std::size_t cs = 256)
        {
            DataFrame::ConfigureDynamic(std::move(h), n, std::move(g), cs);
        }

        void ConfigureStatic(std::vector<std::string> h,
                             std::vector<DataFrameRow> r)
        {
            DataFrame::ConfigureStatic(std::move(h), std::move(r));
        }
    };

    // =========================================================================
    // Measurement stringify / type inspect
    // =========================================================================

    TEST(MeasurementTest, StringifyScalarTypes)
    {
        EXPECT_EQ(Measurement(3.14).to_string(), "3.14");
        EXPECT_EQ(Measurement(42).to_string(), "42");
        EXPECT_EQ(Measurement(std::complex<double>(1.0, -2.0)).to_string(), "1-2i");
        EXPECT_EQ(Measurement(std::string("hello")).to_string(), "hello");
    }

    TEST(MeasurementTest, KindAndDtype)
    {
        EXPECT_EQ(Measurement(3.14).data_type(),    DataType::kReal);
        EXPECT_EQ(Measurement(42).data_type(),      DataType::kInteger);
        EXPECT_EQ(Measurement(std::complex<double>(1,2)).data_type(), DataType::kComplex);
        EXPECT_EQ(Measurement(std::string("s")).data_type(), DataType::kString);

        EXPECT_EQ(Measurement(1.0).data_type(), DataType::kReal);
        EXPECT_NE(Measurement(1.0).data_type(), DataType::kInteger);
        EXPECT_EQ(Measurement(0).data_type(), DataType::kInteger);
        EXPECT_EQ(Measurement(std::complex<double>(0, 0)).data_type(), DataType::kComplex);
        EXPECT_EQ(Measurement(std::string("ok")).data_type(), DataType::kString);
    }

    TEST(MeasurementTest, AsScalar)
    {
        Measurement m(42);
        EXPECT_EQ(m.data_kind(), DataKind::kScalar);
        EXPECT_EQ(m.as_scalar<int>(), 42);

        Measurement m2(std::string("abc"));
        EXPECT_EQ(m2.data_kind(), DataKind::kScalar);
        EXPECT_EQ(m2.as_scalar<std::string>(), "abc");
    }

    // =========================================================================
    // DataFrameRow
    // =========================================================================

    TEST(DataFrameRowTest, FormatMultiIndex)
    {
        DataFrameRow row;
        row.multi_index = {0, 1, 2};
        EXPECT_EQ(row.FormatMultiIndex(), "0,1,2");
    }

    // =========================================================================
    // DataFrame lazy loading & chunking
    // =========================================================================

    TEST(DataFrameLazyTest, RowsLoadedOnFirstAccess)
    {
        int call_count = 0;
        TestDataFrame model;
        model.ConfigureDynamic({"v"}, 100,
            [&](Index start, Index end) -> std::vector<DataFrameRow>
            {
                ++call_count;
                std::vector<DataFrameRow> rows;
                for (Index i = start; i < end; ++i)
                {
                    DataFrameRow r;
                    r.multi_index = {i};
                    r.fields.push_back(Measurement(static_cast<int>(i)));
                    rows.push_back(std::move(r));
                }
                return rows;
            }, 50);

        EXPECT_EQ(call_count, 0);
        EXPECT_EQ(model.GetRow(0).fields[0].to_string(), "0");
        EXPECT_EQ(call_count, 1);           // chunk [0, 50) loaded
        EXPECT_EQ(model.GetRow(49).fields[0].to_string(), "49");
        EXPECT_EQ(call_count, 1);           // same chunk, no reload
        EXPECT_EQ(model.GetRow(50).fields[0].to_string(), "50");
        EXPECT_EQ(call_count, 2);           // chunk [50, 100) loaded
    }

    TEST(DataFrameLazyTest, CacheHitDoesNotReload)
    {
        int call_count = 0;
        TestDataFrame model;
        model.ConfigureDynamic({"v"}, 10,
            [&](Index start, Index end) -> std::vector<DataFrameRow>
            {
                ++call_count;
                std::vector<DataFrameRow> rows;
                for (Index i = start; i < end; ++i)
                {
                    DataFrameRow r;
                    r.multi_index = {i};
                    rows.push_back(std::move(r));
                }
                return rows;
            });

        model.GetRow(0);
        model.GetRow(5);
        model.GetRow(9);
        EXPECT_EQ(call_count, 1);
    }

    TEST(DataFrameLazyTest, MetadataAvailableBeforeLoad)
    {
        TestDataFrame model;
        model.ConfigureDynamic({"x", "y"}, 5,
            [](Index, Index) -> std::vector<DataFrameRow> { return {}; });
        EXPECT_EQ(model.row_count(), 5u);
        ASSERT_EQ(model.headers().size(), 2u);
        EXPECT_EQ(model.headers()[0], "x");
        EXPECT_EQ(model.headers()[1], "y");
    }

    // =========================================================================
    // DataFrame static configuration
    // =========================================================================

    TEST(DataFrameStaticTest, ImmediateAccessAfterConstruction)
    {
        DataFrameRow r1;
        r1.multi_index = {0};
        r1.fields.push_back(Measurement(10));

        DataFrameRow r2;
        r2.multi_index = {1};
        r2.fields.push_back(Measurement(20));

        std::vector<DataFrameRow> rows;
        rows.push_back(r1);
        rows.push_back(r2);

        TestDataFrame model;
        model.ConfigureStatic({"val"}, std::move(rows));

        EXPECT_EQ(model.row_count(), 2u);
        ASSERT_EQ(model.headers().size(), 1u);
        EXPECT_EQ(model.headers()[0], "val");

        // Rows immediately accessible without any generator.
        EXPECT_EQ(model.GetRow(0).fields[0].to_string(), "10");
        EXPECT_EQ(model.GetRow(1).fields[0].to_string(), "20");
    }

    TEST(DataFrameStaticTest, ToCsvFromStatic)
    {
        DataFrameRow r;
        r.multi_index = {42};
        r.fields.push_back(Measurement(3.14));
        r.fields.push_back(Measurement(7));

        std::vector<DataFrameRow> rows;
        rows.push_back(r);

        TestDataFrame model;
        model.ConfigureStatic({"x", "y"}, std::move(rows));

        const std::string csv = model.ToCsv();
        EXPECT_NE(csv.find(",x,y"), std::string::npos);
        EXPECT_NE(csv.find("42,3.14,7"), std::string::npos);
    }

    // =========================================================================
    // DataFrame CSV output
    // =========================================================================

    TEST(DataFrameCsvTest, ToCsvFromBlock)
    {
        Block block(MakeValueRichCreateInfo());
        const DataFrame& table = block.GetOrCreateDataFrame();

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"0,0\",10,1,100"), std::string::npos);
        EXPECT_NE(csv.find("\"1,2\",20,3,105"), std::string::npos);
    }

    TEST(DataFrameCsvTest, ToCsvFromDataArray)
    {
        Block block(MakeRaggedCreateInfo());
        DataArray y = block.GetOrCreateDataArray("y");
        const DataFrame& table = y.GetOrCreateDataFrame();

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y"), std::string::npos);
        EXPECT_NE(csv.find("\"1,1\",20,3"), std::string::npos);
    }

    TEST(DataFrameCsvTest, WriteToCsvRejectsEmptyPath)
    {
        TestDataFrame model;
        model.ConfigureDynamic({"a"}, 1,
            [](Index, Index) -> std::vector<DataFrameRow> { return {}; });
        EXPECT_THROW(model.WriteToCsv(""), std::invalid_argument);
    }
} // namespace xdataset
