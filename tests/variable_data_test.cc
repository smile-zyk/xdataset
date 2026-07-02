#include "../variable_data.h"

#include <gtest/gtest.h>

#include <vector>

using xdataset::Cell;
using xdataset::CellKind;
using xdataset::CellSeries;
using xdataset::Index;
using xdataset::VariableData;

// ---------------------------------------------------------------------------
// VariableData — 1D shape
// ---------------------------------------------------------------------------

TEST(VariableDataTest, Create1DShape) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{5}, 0);
    EXPECT_EQ(vd.rank(), 1u);
    EXPECT_EQ(vd.size(), 5u);
    ASSERT_EQ(vd.shape().size(), 1u);
    EXPECT_EQ(vd.shape()[0], 5);
}

TEST(VariableDataTest, Access1DElement) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{3}, 0);
    vd.at_1d<int>(0) = 10;
    vd.at_1d<int>(1) = 20;
    vd.at_1d<int>(2) = 30;

    EXPECT_EQ(vd.at_1d<int>(0), 10);
    EXPECT_EQ(vd.at_1d<int>(1), 20);
    EXPECT_EQ(vd.at_1d<int>(2), 30);
}

// ---------------------------------------------------------------------------
// VariableData — 2D shape and multi-dimensional indexing
// ---------------------------------------------------------------------------

TEST(VariableDataTest, Create2DShape) {
    VariableData vd = VariableData::Create<double>(std::vector<Index>{3, 5}, 0.0);
    EXPECT_EQ(vd.rank(), 2u);
    EXPECT_EQ(vd.size(), 15u);
    ASSERT_EQ(vd.shape().size(), 2u);
    EXPECT_EQ(vd.shape()[0], 3);
    EXPECT_EQ(vd.shape()[1], 5);
}

TEST(VariableDataTest, Access2DElementByMultiIndex) {
    VariableData vd = VariableData::Create<double>(std::vector<Index>{3, 5}, 0.0);

    // Access via multi-dimensional index: (i, j) -> i*5 + j
    vd.at_2d<double>(0, 1) = 10.5;  // index 0*5 + 1 = 1
    vd.at_2d<double>(1, 2) = 20.3;  // index 1*5 + 2 = 7
    vd.at_2d<double>(2, 4) = 30.7;  // index 2*5 + 4 = 14

    EXPECT_DOUBLE_EQ(vd.at_2d<double>(0, 1), 10.5);
    EXPECT_DOUBLE_EQ(vd.at_2d<double>(1, 2), 20.3);
    EXPECT_DOUBLE_EQ(vd.at_2d<double>(2, 4), 30.7);
}

TEST(VariableDataTest, Access2DElementByFlatIndex) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{3, 5}, 0);

    // Set via flat index
    vd.at_1d<int>(0) = 100;   // (0, 0)
    vd.at_1d<int>(1) = 101;   // (0, 1)
    vd.at_1d<int>(5) = 200;   // (1, 0)
    vd.at_1d<int>(7) = 207;   // (1, 2)
    vd.at_1d<int>(14) = 314;  // (2, 4)

    // Verify via multi-dimensional access
    EXPECT_EQ(vd.at_2d<int>(0, 0), 100);
    EXPECT_EQ(vd.at_2d<int>(0, 1), 101);
    EXPECT_EQ(vd.at_2d<int>(1, 0), 200);
    EXPECT_EQ(vd.at_2d<int>(1, 2), 207);
    EXPECT_EQ(vd.at_2d<int>(2, 4), 314);
}

TEST(VariableDataTest, FlatIndexConversion2D) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{3, 5}, 0);
    std::vector<Index> indices = {1, 2};

    // Direct assignment to verify conversion: (1, 2) -> 1*5 + 2 = 7
    vd.at<int>(indices) = 42;
    EXPECT_EQ(vd.at_1d<int>(7), 42);
}

// ---------------------------------------------------------------------------
// VariableData — 3D shape
// ---------------------------------------------------------------------------

TEST(VariableDataTest, Create3DShape) {
    VariableData vd = VariableData::Create<double>(std::vector<Index>{2, 3, 4}, 0.0);
    EXPECT_EQ(vd.rank(), 3u);
    EXPECT_EQ(vd.size(), 24u);  // 2*3*4
    ASSERT_EQ(vd.shape().size(), 3u);
    EXPECT_EQ(vd.shape()[0], 2);
    EXPECT_EQ(vd.shape()[1], 3);
    EXPECT_EQ(vd.shape()[2], 4);
}

TEST(VariableDataTest, Access3DElement) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{2, 3, 4}, 0);

    // Access via 3D indices: (i, j, k) -> i*(3*4) + j*4 + k
    vd.at_3d<int>(0, 0, 0) = 1;      // 0
    vd.at_3d<int>(0, 1, 2) = 6;      // 0*12 + 1*4 + 2 = 6
    vd.at_3d<int>(1, 2, 3) = 23;     // 1*12 + 2*4 + 3 = 23

    EXPECT_EQ(vd.at_3d<int>(0, 0, 0), 1);
    EXPECT_EQ(vd.at_3d<int>(0, 1, 2), 6);
    EXPECT_EQ(vd.at_3d<int>(1, 2, 3), 23);
}

// ---------------------------------------------------------------------------
// VariableData — exceptions
// ---------------------------------------------------------------------------

TEST(VariableDataTest, ShapeMismatchThrows) {
    CellSeries s = CellSeries::Scalars<int>(10, 0);
    std::vector<Index> shape = {3, 5};  // 15 elements, but series has 10
    EXPECT_THROW({
        VariableData vd(shape, s);
    }, std::invalid_argument);
}

TEST(VariableDataTest, IndexCountMismatchThrows) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{3, 5}, 0);
    std::vector<Index> indices = {1};  // Only 1 index, but shape has rank 2
    EXPECT_THROW(vd.at<int>(indices), std::invalid_argument);
}

TEST(VariableDataTest, IndexOutOfBoundsThrows) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{3, 5}, 0);
    std::vector<Index> indices = {3, 0};  // i=3 out of bounds [0,3)
    EXPECT_THROW(vd.at<int>(indices), std::out_of_range);
}

TEST(VariableDataTest, Access2DOutOfBoundsThrows) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{3, 5}, 0);
    EXPECT_THROW(vd.at_2d<int>(2, 5), std::out_of_range);  // j=5 out of bounds
    EXPECT_THROW(vd.at_2d<int>(3, 0), std::out_of_range);  // i=3 out of bounds
}

// ---------------------------------------------------------------------------
// VariableData — fill and iterate
// ---------------------------------------------------------------------------

TEST(VariableDataTest, FillAllElements2D) {
    VariableData vd = VariableData::Create<double>(std::vector<Index>{2, 3}, 0.0);
    vd.series().fill<double>(5.5);

    for (Index i = 0; i < 2; ++i)
        for (Index j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(vd.at_2d<double>(i, j), 5.5);
}

TEST(VariableDataTest, CreateWithInitialValues) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{2, 3}, 42);

    for (Index i = 0; i < 2; ++i)
        for (Index j = 0; j < 3; ++j)
            EXPECT_EQ(vd.at_2d<int>(i, j), 42);
}

// ---------------------------------------------------------------------------
// VariableData — underlying series access
// ---------------------------------------------------------------------------

TEST(VariableDataTest, AccessUnderlyingSeries) {
    VariableData vd = VariableData::Create<double>(std::vector<Index>{2, 3}, 0.0);
    vd.at_2d<double>(0, 1) = 7.5;

    const CellSeries& s = vd.series();
    EXPECT_EQ(s.size(), 6u);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(1), 7.5);
}

TEST(VariableDataTest, MutableSeriesAccess) {
    VariableData vd = VariableData::Create<int>(std::vector<Index>{2, 3}, 0);
    vd.series().scalar_at<int>(3) = 99;

    // Verify via multi-dimensional access: index 3 = (1, 0)
    EXPECT_EQ(vd.at_2d<int>(1, 0), 99);
}
