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

        void Configure(std::vector<std::string> h, std::size_t n,
                       RowGenerator g, std::size_t cs = 256)
        {
            DataFrame::Configure(std::move(h), n, std::move(g), cs);
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
        EXPECT_EQ(Measurement(3.14).dtype(),    DTypeTag::kReal);
        EXPECT_EQ(Measurement(42).dtype(),      DTypeTag::kInteger);
        EXPECT_EQ(Measurement(std::complex<double>(1,2)).dtype(), DTypeTag::kComplex);
        EXPECT_EQ(Measurement(std::string("s")).dtype(), DTypeTag::kString);

        EXPECT_EQ(Measurement(1.0).dtype(), DTypeTag::kReal);
        EXPECT_NE(Measurement(1.0).dtype(), DTypeTag::kInteger);
        EXPECT_EQ(Measurement(0).dtype(), DTypeTag::kInteger);
        EXPECT_EQ(Measurement(std::complex<double>(0, 0)).dtype(), DTypeTag::kComplex);
        EXPECT_EQ(Measurement(std::string("ok")).dtype(), DTypeTag::kString);
    }

    TEST(MeasurementTest, AsScalar)
    {
        Measurement m(42);
        EXPECT_EQ(m.kind(), DataKind::kScalar);
        EXPECT_EQ(m.as_scalar<int>(), 42);

        Measurement m2(std::string("abc"));
        EXPECT_EQ(m2.kind(), DataKind::kScalar);
        EXPECT_EQ(m2.as_scalar<std::string>(), "abc");
    }

    // =========================================================================
    // DataFrameRow
    // =========================================================================

    TEST(DataFrameRowTest, FormatMultiIndex)
    {
        DataFrameRow row;
        row.multi_index = {0, 1, 2};
        EXPECT_EQ(row.FormatMultiIndex(), "[0,1,2]");
    }

    // =========================================================================
    // DataFrame lazy loading & chunking
    // =========================================================================

    TEST(DataFrameLazyTest, RowsLoadedOnFirstAccess)
    {
        int call_count = 0;
        TestDataFrame model;
        model.Configure({"v"}, 100,
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
        model.Configure({"v"}, 10,
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
        model.Configure({"x", "y"}, 5,
            [](Index, Index) -> std::vector<DataFrameRow> { return {}; });
        EXPECT_EQ(model.row_count(), 5u);
        ASSERT_EQ(model.headers().size(), 2u);
        EXPECT_EQ(model.headers()[0], "x");
        EXPECT_EQ(model.headers()[1], "y");
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
        EXPECT_NE(csv.find("\"[0,0]\",10,1,100"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,2]\",20,3,105"), std::string::npos);
    }

    TEST(DataFrameCsvTest, ToCsvFromDataArray)
    {
        Block block(MakeRaggedCreateInfo());
        DataArray y = block.GetOrCreateDataArray("y");
        const DataFrame& table = y.GetOrCreateDataFrame();

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3"), std::string::npos);
    }

    TEST(DataFrameCsvTest, WriteToCsvRejectsEmptyPath)
    {
        TestDataFrame model;
        model.Configure({"a"}, 1,
            [](Index, Index) -> std::vector<DataFrameRow> { return {}; });
        EXPECT_THROW(model.WriteToCsv(""), std::invalid_argument);
    }
} // namespace xdataset
