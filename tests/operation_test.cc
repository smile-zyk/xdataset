// =============================================================================
//  xdataset -- operation_test.cc
// =============================================================================
//
//  Tests for derive callbacks and the Operate framework.

#include "operation.h"

#include <gtest/gtest.h>

#include <vector>

using xdataset::DataKind;
using xdataset::DataType;
using xdataset::ExecContextInfo;
using xdataset::Index;
using xdataset::OpCategory;
using xdataset::Operate;
using xdataset::OpTraits;
using xdataset::Unit;
using xdataset::Value;
using xdataset::kOpAdd;
using xdataset::kOpSub;

// =========================================================================
//  Derive callbacks
// =========================================================================

TEST(DeriveTest, ShapeBroadcastScalarScalar)
{
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{},
        xdataset::DataShape{}
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd);
    EXPECT_EQ(ds.kind(), DataKind::kScalar);
    EXPECT_TRUE(ds.empty());
}

TEST(DeriveTest, ShapeBroadcastScalarVector)
{
    std::vector<Index> vshape = {5};
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{},
        xdataset::DataShape(vshape)
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd);
    EXPECT_EQ(ds.kind(), DataKind::kVector);
    EXPECT_EQ(ds.dims, vshape);
}

TEST(DeriveTest, ShapeBroadcastVectorVectorSame)
{
    std::vector<Index> s = {3};
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape(s),
        xdataset::DataShape(s)
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kMul);
    EXPECT_EQ(ds.kind(), DataKind::kVector);
    EXPECT_EQ(ds.dims, s);
}

TEST(DeriveTest, ShapeBroadcastVectorVectorMismatchThrows)
{
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{3},
        xdataset::DataShape{4}
    };
    EXPECT_THROW(xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd), std::invalid_argument);
}

TEST(DeriveTest, ShapeBroadcastVectorMatrixThrows)
{
    // Vector [5] x Matrix [2,2] -> col dim mismatch (5 vs 2)
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{5},
        xdataset::DataShape{2, 2}
    };
    EXPECT_THROW(xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd), std::invalid_argument);
}

TEST(DeriveTest, ShapeBroadcastVectorMatrixOk)
{
    // Vector [2] x Matrix [3,2] -> row broadcast (1->3), col matches
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{2},
        xdataset::DataShape{3, 2}
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd);
    EXPECT_EQ(ds.kind(), DataKind::kMatrix);
    EXPECT_EQ(ds.dims, std::vector<Index>({3, 2}));
}

TEST(DeriveTest, ShapeBroadcastMatrixRowBroadcast)
{
    // Matrix [1, 4] x Matrix [3, 4] -> row broadcast (1->3), col matches
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{1, 4},
        xdataset::DataShape{3, 4}
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd);
    EXPECT_EQ(ds.kind(), DataKind::kMatrix);
    EXPECT_EQ(ds.dims, std::vector<Index>({3, 4}));
}

TEST(DeriveTest, ShapeBroadcastMatrixColBroadcast)
{
    // Matrix [3, 1] x Matrix [3, 5] -> col broadcast (1->5), row matches
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{3, 1},
        xdataset::DataShape{3, 5}
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd);
    EXPECT_EQ(ds.kind(), DataKind::kMatrix);
    EXPECT_EQ(ds.dims, std::vector<Index>({3, 5}));
}

TEST(DeriveTest, ShapeBroadcastMatrixRowColMismatchThrows)
{
    // Matrix [2, 3] x Matrix [4, 3] -> both >1 and differ
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{2, 3},
        xdataset::DataShape{4, 3}
    };
    EXPECT_THROW(xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd), std::invalid_argument);
}

TEST(DeriveTest, ShapeBroadcastVectorVectorWidthOneBroadcast)
{
    // Vector [1] x Vector [5] -> col broadcast (1->5)
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{1},
        xdataset::DataShape{5}
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd);
    EXPECT_EQ(ds.kind(), DataKind::kVector);
    EXPECT_EQ(ds.dims, std::vector<Index>({5}));
}

TEST(DeriveTest, ShapeConcatScalars)
{
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{},
        xdataset::DataShape{},
        xdataset::DataShape{}
    };
    auto ds = xdataset::DeriveShapeConcat(ops, OpCategory::kConcat);
    EXPECT_EQ(ds.kind(), DataKind::kVector);
    EXPECT_EQ(ds.dims, std::vector<Index>({3}));
}

TEST(DeriveTest, ShapeConcatVectors)
{
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{4},
        xdataset::DataShape{4}
    };
    auto ds = xdataset::DeriveShapeConcat(ops, OpCategory::kConcat);
    EXPECT_EQ(ds.kind(), DataKind::kMatrix);
    EXPECT_EQ(ds.dims, std::vector<Index>({2, 4}));
}

TEST(DeriveTest, ShapeConcatKindMismatchThrows)
{
    std::vector<xdataset::DataShape> ops = {
        xdataset::DataShape{},
        xdataset::DataShape{5}
    };
    EXPECT_THROW(xdataset::DeriveShapeConcat(ops, OpCategory::kConcat), std::invalid_argument);
}

TEST(DeriveTest, DtypePromote)
{
    std::vector<DataType> dt = {DataType::kInteger, DataType::kReal};
    EXPECT_EQ(xdataset::DeriveDtypePromote(dt, OpCategory::kAdd), DataType::kReal);
}

TEST(DeriveTest, DtypePromoteComplex)
{
    std::vector<DataType> dt = {DataType::kInteger, DataType::kReal, DataType::kComplex};
    EXPECT_EQ(xdataset::DeriveDtypePromote(dt, OpCategory::kAdd), DataType::kComplex);
}

TEST(DeriveTest, DtypeDivIntInt)
{
    std::vector<DataType> dt = {DataType::kInteger, DataType::kInteger};
    EXPECT_EQ(xdataset::DeriveDtypeDiv(dt, OpCategory::kDiv), DataType::kReal);
}

TEST(DeriveTest, DtypeForceInt)
{
    std::vector<DataType> dt = {DataType::kReal, DataType::kComplex};
    EXPECT_EQ(xdataset::DeriveDtypeForceInt(dt, OpCategory::kEq), DataType::kInteger);
}

TEST(DeriveTest, RowsBroadcast)
{
    std::vector<Index> rows = {3, 3};
    EXPECT_EQ(xdataset::DeriveRowsBroadcast(rows, OpCategory::kAdd), 3);
}

TEST(DeriveTest, RowsBroadcastWithOne)
{
    std::vector<Index> rows = {3, 1, 3};
    EXPECT_EQ(xdataset::DeriveRowsBroadcast(rows, OpCategory::kAdd), 3);
}

TEST(DeriveTest, RowsBroadcastAllOne)
{
    std::vector<Index> rows = {1, 1, 1};
    EXPECT_EQ(xdataset::DeriveRowsBroadcast(rows, OpCategory::kAdd), 1);
}

TEST(DeriveTest, RowsBroadcastMismatchThrows)
{
    std::vector<Index> rows = {3, 5};
    EXPECT_THROW(xdataset::DeriveRowsBroadcast(rows, OpCategory::kAdd), std::invalid_argument);
}

// =========================================================================
//  Operate framework -- derivation pipeline tests
// =========================================================================

namespace {

xdataset::Value dummy_execute(const ExecContextInfo& info,
                               const std::vector<Value>& /*ops*/) {
    int tag = static_cast<int>(info.shape.kind()) * 100
            + static_cast<int>(info.dtype) * 10
            + static_cast<int>(info.rows);
    return Value(xdataset::Measurement(tag));
}

const OpTraits kTestOp = {
    OpCategory::kAdd, 2,
    xdataset::DeriveShapeBroadcast,
    xdataset::DeriveRowsBroadcast,
    xdataset::DeriveDtypePromote,
    xdataset::DeriveUnitSameDim,
    dummy_execute
};

}  // anonymous namespace

TEST(OperateTest, TwoMeasurements)
{
    Eigen::RowVectorXd ev(3); ev << 1, 1, 1;
    Value v1(xdataset::Measurement::Vector(ev));
    Value v2(xdataset::Measurement::Vector(ev));
    Value result = Operate({v1, v2}, kTestOp);

    ASSERT_TRUE(result.is_meas());
    int tag = result.as_meas().as_scalar<int>();
    EXPECT_EQ(tag, 101);   // kind=Vector(1)*100 + dtype=Real(0)*10 + rows=1
}

TEST(OperateTest, MeasAndArray)
{
    Value v1(xdataset::Measurement(2.0));

    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    auto da = xdataset::DataArray::CreateIndependent(std::move(ds));
    Value v2(da);

    Value result = Operate({v1, v2}, kTestOp);

    ASSERT_TRUE(result.is_meas());
    int tag = result.as_meas().as_scalar<int>();
    EXPECT_EQ(tag, 3);    // kind=Scalar(0)*100 + dtype=Real(0)*10 + rows=3
}

TEST(OperateTest, ArityMismatchThrows)
{
    Value v1(xdataset::Measurement(1.0));
    EXPECT_THROW(Operate({v1}, kTestOp), std::invalid_argument);         // 1 vs 2
    EXPECT_THROW(Operate({v1, v1, v1}, kTestOp), std::invalid_argument); // 3 vs 2
}

// =========================================================================
//  Operate framework -- execution tests (kOpAdd / kOpSub)
// =========================================================================

#define EXPECT_MEAS_SCALAR_DOUBLE(val, expected) do { \
    ASSERT_TRUE((val).is_meas()); \
    EXPECT_DOUBLE_EQ((val).as_meas().as_scalar<double>(), (expected)); \
} while(0)

#define EXPECT_ARRAY_ELEMENT(row, col, arr, expected) do { \
    ASSERT_LT((row), (arr).as_array().data().size()); \
    EXPECT_DOUBLE_EQ((arr).as_array().data().vector_at<double>((row))((col)), (expected)); \
} while(0)

// ---- Meas x Meas --------------------------------------------------------

TEST(OpAddTest, MeasMeasScalarScalar)
{
    Value v1(xdataset::Measurement(3.0));
    Value v2(xdataset::Measurement(4.0));
    Value result = Operate({v1, v2}, kOpAdd);
    EXPECT_MEAS_SCALAR_DOUBLE(result, 7.0);
}

TEST(OpSubTest, MeasMeasScalarScalar)
{
    Value v1(xdataset::Measurement(10.0));
    Value v2(xdataset::Measurement(3.0));
    Value result = Operate({v1, v2}, kOpSub);
    EXPECT_MEAS_SCALAR_DOUBLE(result, 7.0);
}

TEST(OpAddTest, MeasMeasScalarVectorBroadcast)
{
    Value v1(xdataset::Measurement(2.0));
    Eigen::RowVectorXd ev(3); ev << 1.0, 2.0, 3.0;
    Value v2(xdataset::Measurement::Vector(ev));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 3.0);
    EXPECT_DOUBLE_EQ(vec(1), 4.0);
    EXPECT_DOUBLE_EQ(vec(2), 5.0);
}

TEST(OpAddTest, MeasMeasVectorVector)
{
    Eigen::RowVectorXd a(3); a << 1.0, 2.0, 3.0;
    Eigen::RowVectorXd b(3); b << 4.0, 5.0, 6.0;
    Value result = Operate({Value(xdataset::Measurement::Vector(a)),
                             Value(xdataset::Measurement::Vector(b))}, kOpAdd);
    auto vec = result.as_meas().as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 5.0);
    EXPECT_DOUBLE_EQ(vec(1), 7.0);
    EXPECT_DOUBLE_EQ(vec(2), 9.0);
}

TEST(OpAddTest, MeasMeasMatrixMatrix)
{
    xdataset::MatrixXRd a(2, 2); a << 1.0, 2.0, 3.0, 4.0;
    xdataset::MatrixXRd b(2, 2); b << 5.0, 6.0, 7.0, 8.0;
    Value result = Operate({Value(xdataset::Measurement::Matrix(a)),
                             Value(xdataset::Measurement::Matrix(b))}, kOpAdd);
    auto mat = result.as_meas().as_matrix<double>();
    EXPECT_DOUBLE_EQ(mat(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(mat(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(mat(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 12.0);
}

TEST(OpAddTest, MeasMeasVectorMatrixRowBroadcast)
{
    Eigen::RowVectorXd v(2); v << 10.0, 20.0;
    xdataset::MatrixXRd m(3, 2); m << 1.0, 1.0, 2.0, 2.0, 3.0, 3.0;
    Value result = Operate({Value(xdataset::Measurement::Vector(v)),
                             Value(xdataset::Measurement::Matrix(m))}, kOpAdd);
    auto mat = result.as_meas().as_matrix<double>();
    EXPECT_DOUBLE_EQ(mat(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(mat(0, 1), 21.0);
    EXPECT_DOUBLE_EQ(mat(1, 0), 12.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 22.0);
    EXPECT_DOUBLE_EQ(mat(2, 0), 13.0);
    EXPECT_DOUBLE_EQ(mat(2, 1), 23.0);
}

// ---- Meas x Array -------------------------------------------------------

TEST(OpAddTest, MeasScalarArrayScalar)
{
    Value v1(xdataset::Measurement(5.0));
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 6.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 7.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 8.0);
}

TEST(OpSubTest, MeasScalarArrayScalar)
{
    Value v1(xdataset::Measurement(10.0));
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value result = Operate({v1, v2}, kOpSub);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 9.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 8.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 7.0);
}

TEST(OpAddTest, ArrayScalarMeasScalar)
{
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value v2(xdataset::Measurement(5.0));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 6.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 7.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 8.0);
}

TEST(OpAddTest, MeasScalarArrayVector)
{
    Value v1(xdataset::Measurement(1.0));
    auto ds = xdataset::DataSeries::CreateVector<double>(3, 2);
    ds.vector_at<double>(0) << 10.0, 20.0, 30.0;
    ds.vector_at<double>(1) << 40.0, 50.0, 60.0;
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(0), 11.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(1), 21.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(2), 31.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(1)(0), 41.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(1)(1), 51.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(1)(2), 61.0);
}

// ---- Array x Array (same rows) ------------------------------------------

TEST(OpAddTest, ArrayArrayScalarSameRows)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({10.0, 20.0, 30.0});
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 11.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 22.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 33.0);
}

TEST(OpSubTest, ArrayArrayScalarSameRows)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({10.0, 20.0, 30.0});
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = Operate({v1, v2}, kOpSub);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 9.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 18.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 27.0);
}

TEST(OpAddTest, ArrayArrayVectorSameRowsSameShape)
{
    auto ds1 = xdataset::DataSeries::CreateVector<double>(2, 3);
    ds1.vector_at<double>(0) << 1.0, 2.0;
    ds1.vector_at<double>(1) << 3.0, 4.0;
    ds1.vector_at<double>(2) << 5.0, 6.0;
    auto ds2 = xdataset::DataSeries::CreateVector<double>(2, 3);
    ds2.vector_at<double>(0) << 10.0, 20.0;
    ds2.vector_at<double>(1) << 30.0, 40.0;
    ds2.vector_at<double>(2) << 50.0, 60.0;
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(0), 11.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(1), 22.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(1)(0), 33.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(1)(1), 44.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(2)(0), 55.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(2)(1), 66.0);
}

// ---- Array x Array (row broadcast) --------------------------------------

TEST(OpAddTest, ArrayArrayScalarBroadcastRows)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0});  // 1 row
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({2.0, 4.0, 6.0}); // 3 rows
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.size(), 3u);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 3.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 5.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 7.0);
}

TEST(OpAddTest, ArrayArrayVectorBroadcastRows)
{
    auto ds1 = xdataset::DataSeries::CreateVector<double>(2, 1);
    ds1.vector_at<double>(0) << 1.0, 2.0;  // 1 row
    auto ds2 = xdataset::DataSeries::CreateVector<double>(2, 3);
    ds2.vector_at<double>(0) << 10.0, 20.0;
    ds2.vector_at<double>(1) << 30.0, 40.0;
    ds2.vector_at<double>(2) << 50.0, 60.0;
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.size(), 3u);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(0), 11.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(1), 22.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(1)(0), 31.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(1)(1), 42.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(2)(0), 51.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(2)(1), 62.0);
}

// ---- Meas x Meas int -> double promotion -------------------------------

TEST(OpAddTest, MeasMeasIntAndRealPromoteToReal)
{
    Value v1(xdataset::Measurement(3));    // int
    Value v2(xdataset::Measurement(4.5));  // real
    Value result = Operate({v1, v2}, kOpAdd);
    EXPECT_MEAS_SCALAR_DOUBLE(result, 7.5);
}

// ---- Meas x Array int -> double promotion via FlatInput ----------------

TEST(OpAddTest, MeasIntArrayReal)
{
    Value v1(xdataset::Measurement(3));  // int scalar
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value result = Operate({v1, v2}, kOpAdd);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 5.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 6.0);
}
