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

// =========================================================================
//  Derive callbacks
// =========================================================================

TEST(DeriveTest, ShapeBroadcastScalarScalar)
{
    std::vector<xdataset::ShapeInfo> ops = {
        {DataKind::kScalar, {}},
        {DataKind::kScalar, {}}
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd);
    EXPECT_EQ(ds.kind, DataKind::kScalar);
    EXPECT_TRUE(ds.shape.empty());
}

TEST(DeriveTest, ShapeBroadcastScalarVector)
{
    std::vector<Index> vshape = {5};
    std::vector<xdataset::ShapeInfo> ops = {
        {DataKind::kScalar, {}},
        {DataKind::kVector, vshape}
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd);
    EXPECT_EQ(ds.kind, DataKind::kVector);
    EXPECT_EQ(ds.shape, vshape);
}

TEST(DeriveTest, ShapeBroadcastVectorVectorSame)
{
    std::vector<Index> s = {3};
    std::vector<xdataset::ShapeInfo> ops = {
        {DataKind::kVector, s},
        {DataKind::kVector, s}
    };
    auto ds = xdataset::DeriveShapeBroadcast(ops, OpCategory::kMul);
    EXPECT_EQ(ds.kind, DataKind::kVector);
    EXPECT_EQ(ds.shape, s);
}

TEST(DeriveTest, ShapeBroadcastVectorVectorMismatchThrows)
{
    std::vector<xdataset::ShapeInfo> ops = {
        {DataKind::kVector, {3}},
        {DataKind::kVector, {4}}
    };
    EXPECT_THROW(xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd), std::invalid_argument);
}

TEST(DeriveTest, ShapeBroadcastVectorMatrixThrows)
{
    std::vector<xdataset::ShapeInfo> ops = {
        {DataKind::kVector, {5}},
        {DataKind::kMatrix, {2, 2}}
    };
    EXPECT_THROW(xdataset::DeriveShapeBroadcast(ops, OpCategory::kAdd), std::invalid_argument);
}

TEST(DeriveTest, ShapeConcatScalars)
{
    std::vector<xdataset::ShapeInfo> ops = {
        {DataKind::kScalar, {}},
        {DataKind::kScalar, {}},
        {DataKind::kScalar, {}}
    };
    auto ds = xdataset::DeriveShapeConcat(ops, OpCategory::kConcat);
    EXPECT_EQ(ds.kind, DataKind::kVector);
    EXPECT_EQ(ds.shape, std::vector<Index>({3}));
}

TEST(DeriveTest, ShapeConcatVectors)
{
    std::vector<xdataset::ShapeInfo> ops = {
        {DataKind::kVector, {4}},
        {DataKind::kVector, {4}}
    };
    auto ds = xdataset::DeriveShapeConcat(ops, OpCategory::kConcat);
    EXPECT_EQ(ds.kind, DataKind::kMatrix);
    EXPECT_EQ(ds.shape, std::vector<Index>({2, 4}));
}

TEST(DeriveTest, ShapeConcatKindMismatchThrows)
{
    std::vector<xdataset::ShapeInfo> ops = {
        {DataKind::kScalar, {}},
        {DataKind::kVector, {3}}
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
//  Operate framework
// =========================================================================

namespace {

xdataset::Value dummy_execute(const ExecContextInfo& info,
                               const std::vector<Value>& /*ops*/) {
    int tag = static_cast<int>(info.shape.kind) * 100
            + static_cast<int>(info.dtype) * 10
            + static_cast<int>(info.rows);
    return Value(xdataset::Measurement(tag));
}

const OpTraits kTestOp = {
    OpCategory::kAdd, 2,
    xdataset::DeriveShapeBroadcast,
    xdataset::DeriveRowsBroadcast,
    xdataset::DeriveDtypePromote,
    xdataset::DeriveUnitAdd,
    dummy_execute
};

}  // anonymous namespace

TEST(OperateTest, TwoMeasurements)
{
    Eigen::VectorXd ev(3); ev << 1, 1, 1;
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
