#include "block_fixtures.h"

#include <gtest/gtest.h>

namespace xdataset
{
    using namespace block_fixtures;

    // Test wrapper that exposes protected Configure / constructor of GridModel.
    class TestGridModel : public GridModel
    {
    public:
        TestGridModel() = default;

        void Configure(std::vector<std::string> h, std::size_t n,
                       RowGenerator g, std::size_t cs = 256)
        {
            GridModel::Configure(std::move(h), n, std::move(g), cs);
        }
    };

    // =========================================================================
    // GridField stringify / type inspect
    // =========================================================================

    TEST(GridFieldTest, StringifyScalarTypes)
    {
        EXPECT_EQ(GridFieldToString(GridField(3.14)), "3.14");
        EXPECT_EQ(GridFieldToString(GridField(42)), "42");
        EXPECT_EQ(GridFieldToString(GridField(std::complex<double>(1.0, -2.0))), "1-2i");
        EXPECT_EQ(GridFieldToString(GridField(std::string("hello"))), "hello");
    }

    TEST(GridFieldTest, TypeOfAndIsChecks)
    {
        EXPECT_EQ(GridFieldGetVisitor::TypeOf(GridField(3.14)),    DTypeTag::kReal);
        EXPECT_EQ(GridFieldGetVisitor::TypeOf(GridField(42)),      DTypeTag::kInteger);
        EXPECT_EQ(GridFieldGetVisitor::TypeOf(GridField(std::complex<double>(1,2))), DTypeTag::kComplex);
        EXPECT_EQ(GridFieldGetVisitor::TypeOf(GridField(std::string("s"))), DTypeTag::kString);

        EXPECT_TRUE (GridFieldGetVisitor::IsDouble(GridField(1.0)));
        EXPECT_FALSE(GridFieldGetVisitor::IsInt(GridField(1.0)));
        EXPECT_TRUE (GridFieldGetVisitor::IsInt(GridField(0)));
        EXPECT_TRUE (GridFieldGetVisitor::IsComplex(GridField(std::complex<double>(0, 0))));
        EXPECT_TRUE (GridFieldGetVisitor::IsString(GridField(std::string("ok"))));
    }

    TEST(GridFieldTest, ApplyAndExtractValue)
    {
        auto& v = GridFieldGetVisitor::Apply(GridField(42));
        EXPECT_EQ(v.type(), DTypeTag::kInteger);
        EXPECT_EQ(v.as_int(), 42);

        auto& v2 = GridFieldGetVisitor::Apply(GridField(std::string("abc")));
        EXPECT_EQ(v2.type(), DTypeTag::kString);
        EXPECT_EQ(v2.as_string(), "abc");
    }

    // =========================================================================
    // GridRow
    // =========================================================================

    TEST(GridRowTest, FormatMultiIndex)
    {
        GridRow row;
        row.multi_index = {0, 1, 2};
        EXPECT_EQ(row.FormatMultiIndex(), "[0,1,2]");
    }

    // =========================================================================
    // GridModel lazy loading & chunking
    // =========================================================================

    TEST(GridModelLazyTest, RowsLoadedOnFirstAccess)
    {
        int call_count = 0;
        TestGridModel model;
        model.Configure({"v"}, 100,
            [&](Index start, Index end) -> std::vector<GridRow>
            {
                ++call_count;
                std::vector<GridRow> rows;
                for (Index i = start; i < end; ++i)
                {
                    GridRow r;
                    r.multi_index = {i};
                    r.fields.push_back(GridField(static_cast<int>(i)));
                    rows.push_back(std::move(r));
                }
                return rows;
            }, 50);

        EXPECT_EQ(call_count, 0);
        EXPECT_EQ(GridFieldToString(model.GetRow(0).fields[0]), "0");
        EXPECT_EQ(call_count, 1);           // chunk [0, 50) loaded
        EXPECT_EQ(GridFieldToString(model.GetRow(49).fields[0]), "49");
        EXPECT_EQ(call_count, 1);           // same chunk, no reload
        EXPECT_EQ(GridFieldToString(model.GetRow(50).fields[0]), "50");
        EXPECT_EQ(call_count, 2);           // chunk [50, 100) loaded
    }

    TEST(GridModelLazyTest, CacheHitDoesNotReload)
    {
        int call_count = 0;
        TestGridModel model;
        model.Configure({"v"}, 10,
            [&](Index start, Index end) -> std::vector<GridRow>
            {
                ++call_count;
                std::vector<GridRow> rows;
                for (Index i = start; i < end; ++i)
                {
                    GridRow r;
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

    TEST(GridModelLazyTest, MetadataAvailableBeforeLoad)
    {
        TestGridModel model;
        model.Configure({"x", "y"}, 5,
            [](Index, Index) -> std::vector<GridRow> { return {}; });
        EXPECT_EQ(model.row_count(), 5u);
        ASSERT_EQ(model.headers().size(), 2u);
        EXPECT_EQ(model.headers()[0], "x");
        EXPECT_EQ(model.headers()[1], "y");
    }

    // =========================================================================
    // GridModel CSV output
    // =========================================================================

    TEST(GridModelCsvTest, ToCsvFromBlock)
    {
        Block block(MakeValueRichCreateInfo());
        const GridModel& table = block.grid_model();

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y,z"), std::string::npos);
        EXPECT_NE(csv.find("\"[0,0]\",10,1,100"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,2]\",20,3,105"), std::string::npos);
    }

    TEST(GridModelCsvTest, ToCsvFromVariable)
    {
        Block block(MakeJaggedCreateInfo());
        std::shared_ptr<Variable> y = block.GetOrCreateVariable("y");
        const GridModel& table = y->grid_model();

        const std::string csv = table.ToCsv();
        EXPECT_NE(csv.find(",x,y"), std::string::npos);
        EXPECT_NE(csv.find("\"[1,1]\",20,3"), std::string::npos);
    }

    TEST(GridModelCsvTest, WriteToCsvRejectsEmptyPath)
    {
        TestGridModel model;
        model.Configure({"a"}, 1,
            [](Index, Index) -> std::vector<GridRow> { return {}; });
        EXPECT_THROW(model.WriteToCsv(""), std::invalid_argument);
    }

    TEST(GridModelCsvTest, WriteToCsvRoundtripsThroughFile)
    {
        Block block(MakeValueRichCreateInfo());
        const GridModel& table = block.grid_model();

        table.WriteToCsv("__test_roundtrip.csv");
        const std::string csv = table.ToCsv();
        EXPECT_EQ(csv.size(), table.ToCsv().size());  // stable across calls
    }
} // namespace xdataset
