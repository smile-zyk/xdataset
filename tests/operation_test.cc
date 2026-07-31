// =============================================================================
//  xdataset -- operation_test.cc
// =============================================================================
//
//  Tests for the public Operation* API.

#include "operation.h"
#include "block_fixtures.h"

#include <gtest/gtest.h>

#include <vector>

using xdataset::Block;
using xdataset::DataArray;
using xdataset::DataArrayKind;
using xdataset::DataKind;
using xdataset::DataType;
using xdataset::Index;
using xdataset::Unit;
using xdataset::Value;
using xdataset::VecXd;
using xdataset::VecXi;
using xdataset::VecXcd;
using xdataset::VecXs;
using xdataset::MatXd;
using xdataset::MatXi;
using xdataset::MatXcd;
using xdataset::MatXs;
using xdataset::block_fixtures::MakeBaseCreateInfo;

using xdataset::OperationAdd;
using xdataset::OperationSub;
using xdataset::OperationMul;
using xdataset::OperationDiv;
using xdataset::OperationMod;
using xdataset::OperationPow;
using xdataset::OperationNegate;
using xdataset::OperationNot;
using xdataset::OperationBitNot;
using xdataset::OperationEq;
using xdataset::OperationNeq;
using xdataset::OperationLt;
using xdataset::OperationGt;
using xdataset::OperationLe;
using xdataset::OperationGe;
using xdataset::OperationAnd;
using xdataset::OperationOr;
using xdataset::OperationBitAnd;
using xdataset::OperationBitOr;
using xdataset::OperationBitXor;
using xdataset::OperationShl;
using xdataset::OperationShr;
using xdataset::OperationMatrix;
using xdataset::OperationSweep;

#define EXPECT_MEAS_SCALAR_DOUBLE(val, expected) do { \
    ASSERT_TRUE((val).is_meas()); \
    EXPECT_DOUBLE_EQ((val).as_meas().as_scalar<double>(), (expected)); \
} while(0)

// =========================================================================
//  OperationAdd / OperationSub
// =========================================================================

TEST(OperationAddTest, MeasMeasScalarScalar)
{
    Value v1(xdataset::Measurement(3.0));
    Value v2(xdataset::Measurement(4.0));
    Value result = OperationAdd(v1, v2);
    EXPECT_MEAS_SCALAR_DOUBLE(result, 7.0);
}

TEST(OperationSubTest, MeasMeasScalarScalar)
{
    Value v1(xdataset::Measurement(10.0));
    Value v2(xdataset::Measurement(3.0));
    Value result = OperationSub(v1, v2);
    EXPECT_MEAS_SCALAR_DOUBLE(result, 7.0);
}

TEST(OperationAddTest, MeasMeasScalarVectorBroadcast)
{
    Value v1(xdataset::Measurement(2.0));
    VecXd ev(3); ev << 1.0, 2.0, 3.0;
    Value v2(xdataset::Measurement::Vector(ev));
    Value result = OperationAdd(v1, v2);
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 3.0);
    EXPECT_DOUBLE_EQ(vec(1), 4.0);
    EXPECT_DOUBLE_EQ(vec(2), 5.0);
}

TEST(OperationAddTest, MeasMeasVectorVector)
{
    VecXd a(3); a << 1.0, 2.0, 3.0;
    VecXd b(3); b << 4.0, 5.0, 6.0;
    Value result = OperationAdd(Value(xdataset::Measurement::Vector(a)),
                                 Value(xdataset::Measurement::Vector(b)));
    auto vec = result.as_meas().as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 5.0);
    EXPECT_DOUBLE_EQ(vec(1), 7.0);
    EXPECT_DOUBLE_EQ(vec(2), 9.0);
}

TEST(OperationAddTest, MeasMeasIntAndRealPromoteToReal)
{
    Value v1(xdataset::Measurement(3));
    Value v2(xdataset::Measurement(4.5));
    Value result = OperationAdd(v1, v2);
    EXPECT_MEAS_SCALAR_DOUBLE(result, 7.5);
}

// ---- Meas x Array -------------------------------------------------------

TEST(OperationAddTest, MeasScalarArrayScalar)
{
    Value v1(xdataset::Measurement(5.0));
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value result = OperationAdd(v1, v2);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 6.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 7.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 8.0);
}

TEST(OperationAddTest, ArrayScalarMeasScalar)
{
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value v2(xdataset::Measurement(5.0));
    Value result = OperationAdd(v1, v2);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 6.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 7.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 8.0);
}

TEST(OperationAddTest, ArrayArrayScalarSameRows)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({10.0, 20.0, 30.0});
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = OperationAdd(v1, v2);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 11.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 22.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 33.0);
}

// =========================================================================
//  OperationMul
// =========================================================================

TEST(OperationMulTest, MeasMeasScalarScalar)
{
    Value result = OperationMul(Value(xdataset::Measurement(3.0)),
                                 Value(xdataset::Measurement(4.0)));
    EXPECT_MEAS_SCALAR_DOUBLE(result, 12.0);
}

TEST(OperationMulTest, MeasMeasVectorxMatrix)
{
    VecXd v(2); v << 1.0, 2.0;
    MatXd m(2, 3);
    m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Value result = OperationMul(Value(xdataset::Measurement::Vector(v)),
                                 Value(xdataset::Measurement::Matrix(m)));
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 9.0);
    EXPECT_DOUBLE_EQ(vec(1), 12.0);
    EXPECT_DOUBLE_EQ(vec(2), 15.0);
}

TEST(OperationMulTest, MatrixMulByScalarIsElementwise)
{
    MatXd m(2, 2); m << 1.0, 2.0, 3.0, 4.0;
    Value result = OperationMul(Value(xdataset::Measurement::Matrix(m)),
                                 Value(xdataset::Measurement(2.0)));
    ASSERT_TRUE(result.is_meas());
    auto mat = result.as_meas().as_matrix<double>();
    EXPECT_DOUBLE_EQ(mat(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 8.0);
}

TEST(OperationMulTest, ArrayArrayScalarSameRows)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({4.0, 5.0});
    Value result = OperationMul(
        Value(xdataset::DataArray::CreateIndependent(std::move(ds1))),
        Value(xdataset::DataArray::CreateIndependent(std::move(ds2))));
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 8.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 15.0);
}

// =========================================================================
//  OperationDiv
// =========================================================================

TEST(OperationDivTest, MeasMeasScalarScalar)
{
    Value result = OperationDiv(Value(xdataset::Measurement(12.0)),
                                 Value(xdataset::Measurement(4.0)));
    EXPECT_MEAS_SCALAR_DOUBLE(result, 3.0);
}

TEST(OperationDivTest, ScalarDivByZeroThrows)
{
    EXPECT_THROW(OperationDiv(Value(xdataset::Measurement(1.0)),
                               Value(xdataset::Measurement(0.0))),
                 std::invalid_argument);
}

TEST(OperationDivTest, MeasMeasVectorDivMatrix)
{
    VecXd v(2); v << 6.0, 8.0;
    MatXd m(2, 2); m << 2.0, 0.0, 0.0, 4.0;
    Value result = OperationDiv(Value(xdataset::Measurement::Vector(v)),
                                 Value(xdataset::Measurement::Matrix(m)));
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 3.0);
    EXPECT_DOUBLE_EQ(vec(1), 2.0);
}

// =========================================================================
//  OperationMod
// =========================================================================

TEST(OperationModTest, MeasMeasIntScalarScalar)
{
    Value result = OperationMod(Value(xdataset::Measurement(10)),
                                 Value(xdataset::Measurement(3)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<int>(), 1);
}

TEST(OperationModTest, MeasMeasDoubleScalarScalar)
{
    Value result = OperationMod(Value(xdataset::Measurement(10.5)),
                                 Value(xdataset::Measurement(3.0)));
    EXPECT_MEAS_SCALAR_DOUBLE(result, 1.5);
}

TEST(OperationModTest, ScalarModByZeroThrows)
{
    EXPECT_THROW(OperationMod(Value(xdataset::Measurement(10)),
                               Value(xdataset::Measurement(0))),
                 std::invalid_argument);
}

// =========================================================================
//  OperationPow
// =========================================================================

TEST(OperationPowTest, MeasMeasScalarScalar)
{
    Value result = OperationPow(Value(xdataset::Measurement(2.0)),
                                 Value(xdataset::Measurement(3.0)));
    EXPECT_MEAS_SCALAR_DOUBLE(result, 8.0);
}

TEST(OperationPowTest, MeasMeasScalarVectorBroadcast)
{
    Value v1(xdataset::Measurement(2.0));
    VecXd ev(3); ev << 1.0, 2.0, 3.0;
    Value v2(xdataset::Measurement::Vector(ev));
    Value result = OperationPow(v1, v2);
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 2.0);
    EXPECT_DOUBLE_EQ(vec(1), 4.0);
    EXPECT_DOUBLE_EQ(vec(2), 8.0);
}

// =========================================================================
//  OperationNegate
// =========================================================================

TEST(OperationNegateTest, MeasScalar)
{
    Value result = OperationNegate(Value(xdataset::Measurement(5.0)));
    EXPECT_MEAS_SCALAR_DOUBLE(result, -5.0);
}

TEST(OperationNegateTest, ArrayScalar)
{
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, -2.0, 3.0});
    Value v(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value result = OperationNegate(v);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), -1.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 2.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), -3.0);
}

// =========================================================================
//  OperationNot
// =========================================================================

TEST(OperationNotTest, MeasScalarZero)
{
    Value result = OperationNot(Value(xdataset::Measurement(0.0)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationNotTest, MeasScalarNonZero)
{
    Value result = OperationNot(Value(xdataset::Measurement(3.5)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);
}

// =========================================================================
//  OperationBitNot
// =========================================================================

TEST(OperationBitNotTest, MeasScalar)
{
    Value result = OperationBitNot(Value(xdataset::Measurement(0)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<int>(), ~0);
}

// =========================================================================
//  OperationEq / OperationNeq / OperationLt / OperationGt
// =========================================================================

TEST(OperationEqTest, MeasMeasScalarEqual)
{
    Value result = OperationEq(Value(xdataset::Measurement(3.0)),
                                Value(xdataset::Measurement(3.0)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationEqTest, MeasMeasScalarNotEqual)
{
    Value result = OperationEq(Value(xdataset::Measurement(3.0)),
                                Value(xdataset::Measurement(4.0)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);
}

TEST(OperationNeqTest, MeasMeasScalar)
{
    Value result = OperationNeq(Value(xdataset::Measurement(3.0)),
                                 Value(xdataset::Measurement(4.0)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationLtTest, MeasMeasScalar)
{
    Value result = OperationLt(Value(xdataset::Measurement(2.0)),
                                Value(xdataset::Measurement(5.0)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationGtTest, MeasMeasScalar)
{
    Value result = OperationGt(Value(xdataset::Measurement(5.0)),
                                Value(xdataset::Measurement(2.0)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

// ---- Cmp: int (eq/ne/lt/gt/le/ge) --------------------------------------

TEST(OperationLtTest, IntScalar)
{
    Value v1(xdataset::Measurement(3));
    Value v2(xdataset::Measurement(7));
    Value result = OperationLt(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationGtTest, IntScalar)
{
    Value v1(xdataset::Measurement(7));
    Value v2(xdataset::Measurement(3));
    Value result = OperationGt(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationLeTest, IntScalar)
{
    Value v1(xdataset::Measurement(3));
    Value v2(xdataset::Measurement(3));
    Value result = OperationLe(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

// ---- Cmp: complex (abs() for < <= > >=) --------------------------------

TEST(OperationEqTest, ComplexEqual)
{
    Value v1(xdataset::Measurement(std::complex<double>(1.0, 2.0)));
    Value v2(xdataset::Measurement(std::complex<double>(1.0, 2.0)));
    Value result = OperationEq(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationNeqTest, ComplexNotEqual)
{
    Value v1(xdataset::Measurement(std::complex<double>(1.0, 2.0)));
    Value v2(xdataset::Measurement(std::complex<double>(3.0, 4.0)));
    Value result = OperationNeq(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationLtTest, ComplexAbs)
{
    // |3+4i| = 5  <  |6+0i| = 6
    Value v1(xdataset::Measurement(std::complex<double>(3.0, 4.0)));
    Value v2(xdataset::Measurement(std::complex<double>(6.0, 0.0)));
    Value result = OperationLt(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationGtTest, ComplexAbs)
{
    // |5+0i| = 5  >  |3+4i| = 5  → false (equal abs)
    Value v1(xdataset::Measurement(std::complex<double>(5.0, 0.0)));
    Value v2(xdataset::Measurement(std::complex<double>(3.0, 4.0)));
    Value result = OperationGt(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);
}

TEST(OperationLeTest, ComplexAbs)
{
    Value v1(xdataset::Measurement(std::complex<double>(3.0, 4.0)));
    Value v2(xdataset::Measurement(std::complex<double>(5.0, 0.0)));
    Value result = OperationLe(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);  // 5 <= 5
}

// ---- Cmp: string -------------------------------------------------------

TEST(OperationEqTest, StringEqual)
{
    Value v1(xdataset::Measurement::String("abc"));
    Value v2(xdataset::Measurement::String("abc"));
    Value result = OperationEq(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationNeqTest, StringNotEqual)
{
    Value v1(xdataset::Measurement::String("abc"));
    Value v2(xdataset::Measurement::String("xyz"));
    Value result = OperationNeq(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationLtTest, StringLex)
{
    Value v1(xdataset::Measurement::String("abc"));
    Value v2(xdataset::Measurement::String("xyz"));
    Value result = OperationLt(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationGtTest, StringLex)
{
    Value v1(xdataset::Measurement::String("xyz"));
    Value v2(xdataset::Measurement::String("abc"));
    Value result = OperationGt(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

// ---- Cmp: mixed types (int↔double compare directly) ---------------------

TEST(OperationEqTest, IntAndDouble)
{
    Value v1(xdataset::Measurement(3));
    Value v2(xdataset::Measurement(3.0));
    Value result = OperationEq(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

TEST(OperationLtTest, IntAndDouble)
{
    Value v1(xdataset::Measurement(3));
    Value v2(xdataset::Measurement(5.5));
    Value result = OperationLt(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

// ---- Cmp: mixed complex↔real (promote to complex, use abs() for < >) -----

TEST(OperationLtTest, ComplexAndReal)
{
    // real 5.0 becomes (5,0), |3+4i|=5, |5+0i|=5 → not less
    Value v1(xdataset::Measurement(std::complex<double>(3.0, 4.0)));
    Value v2(xdataset::Measurement(5.0));
    Value result = OperationLt(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);  // 5 < 5 → false
}

// ---- Cmp: row broadcast (Array x Array) ---------------------------------

TEST(OperationEqTest, ArrayArrayScalarSameRows)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 3.0, 5.0});
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 3.0, 5.0});
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = OperationEq(v1, v2);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.scalar_at<int>(0), 1);
    EXPECT_EQ(arr.scalar_at<int>(1), 1);
    EXPECT_EQ(arr.scalar_at<int>(2), 1);
}

TEST(OperationLtTest, ArrayArrayBroadcastRows)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({3.0});      // 1 row
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 5.0});  // 2 rows
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = OperationLt(v1, v2);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.size(), 2u);
    EXPECT_EQ(arr.scalar_at<int>(0), 0);  // 3 < 1 → 0
    EXPECT_EQ(arr.scalar_at<int>(1), 1);  // 3 < 5 → 1
}

// ---- Cmp: cell broadcast (Vector broadcast in Meas) ---------------------

TEST(OperationEqTest, MeasVectorMeasScalarBroadcast)
{
    VecXd a(3); a << 2.0, 2.0, 2.0;
    Value v1(xdataset::Measurement::Vector(a));
    Value v2(xdataset::Measurement(2.0));
    Value result = OperationEq(v1, v2);
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<int>();
    EXPECT_EQ(vec(0), 1); EXPECT_EQ(vec(1), 1); EXPECT_EQ(vec(2), 1);
}

// ---- Logic: row broadcast (Array x Array) --------------------------------

TEST(OperationAndTest, ArrayArrayBroadcastRows)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0});     // 1 row, non-zero
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({0.0, 3.0}); // 2 rows
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = OperationAnd(v1, v2);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.size(), 2u);
    EXPECT_EQ(arr.scalar_at<int>(0), 0);  // 1 && 0 = 0
    EXPECT_EQ(arr.scalar_at<int>(1), 1);  // 1 && 1 = 1
}

// ---- Not: array ------------------------------------------------

TEST(OperationNotTest, ArrayScalar)
{
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({0.0, 5.0, -3.0});
    Value v(xdataset::DataArray::CreateIndependent(std::move(ds)));
    Value result = OperationNot(v);
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.scalar_at<int>(0), 1);
    EXPECT_EQ(arr.scalar_at<int>(1), 0);
    EXPECT_EQ(arr.scalar_at<int>(2), 0);
}

// =========================================================================
//  OperationAnd / OperationOr
// =========================================================================

TEST(OperationAndTest, MeasMeasScalarScalar)
{
    Value v1(xdataset::Measurement(0.0));
    Value v2(xdataset::Measurement(1.0));
    Value result = OperationAnd(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);
}

TEST(OperationAndTest, BoolOperands)
{
    Value v1(xdataset::Measurement::Boolean(true));
    Value v2(xdataset::Measurement::Boolean(false));
    Value result = OperationAnd(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);
}

TEST(OperationOrTest, BoolOperands)
{
    Value v1(xdataset::Measurement::Boolean(true));
    Value v2(xdataset::Measurement::Boolean(false));
    Value result = OperationOr(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

// ---- Logic: real/complex operands (as_logical: non-zero→1) --------------

TEST(OperationAndTest, RealOperands)
{
    Value v1(xdataset::Measurement(3.5));
    Value v2(xdataset::Measurement(0.0));
    Value result = OperationAnd(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);  // 1 && 0 = 0
}

TEST(OperationOrTest, RealOperands)
{
    Value v1(xdataset::Measurement(0.0));
    Value v2(xdataset::Measurement(-2.0));
    Value result = OperationOr(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);  // 0 || 1 = 1
}

TEST(OperationAndTest, ComplexOperands)
{
    // (1+0i) non-zero → 1, (0+0i) zero → 0
    Value v1(xdataset::Measurement(std::complex<double>(1.0, 0.0)));
    Value v2(xdataset::Measurement(std::complex<double>(0.0, 0.0)));
    Value result = OperationAnd(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);  // 1 && 0 = 0
}

TEST(OperationOrTest, ComplexOperands)
{
    // (0+5i) non-zero → 1
    Value v1(xdataset::Measurement(std::complex<double>(0.0, 5.0)));
    Value v2(xdataset::Measurement(std::complex<double>(0.0, 0.0)));
    Value result = OperationOr(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);  // 1 || 0 = 1
}

// ---- Logic: vector operands (as_logical broadcasts) ---------------------

TEST(OperationAndTest, VectorOperands)
{
    VecXd a(2); a << 0.0, 1.0;
    VecXd b(2); b << 2.0, 0.0;
    Value v1(xdataset::Measurement::Vector(a));
    Value v2(xdataset::Measurement::Vector(b));
    Value result = OperationAnd(v1, v2);
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<int>();
    EXPECT_EQ(vec(0), 0);  // 0 && 1
    EXPECT_EQ(vec(1), 0);  // 1 && 0
}

// ---- Logic: Not with real/complex --------------------------------------

TEST(OperationNotTest, RealOperand)
{
    Value result = OperationNot(Value(xdataset::Measurement(3.5)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);  // !1 = 0
}

TEST(OperationNotTest, ComplexOperand)
{
    // (0+5i) non-zero → !1 = 0
    Value result = OperationNot(Value(xdataset::Measurement(std::complex<double>(0.0, 5.0))));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), false);
}

TEST(OperationNotTest, BoolOperand)
{
    Value result = OperationNot(Value(xdataset::Measurement::Boolean(false)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<bool>(), true);
}

// =========================================================================
//  OperationBitAnd / OperationBitOr / OperationBitXor
// =========================================================================

TEST(OperationBitAndTest, MeasMeasScalar)
{
    Value v1(xdataset::Measurement(6));   // 110
    Value v2(xdataset::Measurement(3));   // 011
    Value result = OperationBitAnd(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<int>(), 2);  // 010
}

TEST(OperationBitOrTest, MeasMeasScalar)
{
    Value v1(xdataset::Measurement(6));
    Value v2(xdataset::Measurement(3));
    Value result = OperationBitOr(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<int>(), 7);
}

TEST(OperationBitXorTest, MeasMeasScalar)
{
    Value v1(xdataset::Measurement(6));
    Value v2(xdataset::Measurement(3));
    Value result = OperationBitXor(v1, v2);
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<int>(), 5);
}

// =========================================================================
//  OperationShl / OperationShr
// =========================================================================

TEST(OperationShlTest, MeasMeasScalar)
{
    Value result = OperationShl(Value(xdataset::Measurement(1)),
                                 Value(xdataset::Measurement(3)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<int>(), 8);
}

TEST(OperationShrTest, MeasMeasScalar)
{
    Value result = OperationShr(Value(xdataset::Measurement(16)),
                                 Value(xdataset::Measurement(2)));
    ASSERT_TRUE(result.is_meas());
    EXPECT_EQ(result.as_meas().as_scalar<int>(), 4);
}

// =========================================================================
//  OperationMatrix
// =========================================================================

TEST(OperationMatrixTest, IntAndRealPromote)
{
    Value result = OperationMatrix({Value(xdataset::Measurement(1)),
                                     Value(xdataset::Measurement(2.5))});
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 1.0);
    EXPECT_DOUBLE_EQ(vec(1), 2.5);
}

TEST(OperationMatrixTest, StringScalarsToVector)
{
    Value v1(xdataset::Measurement::String("hello"));
    Value v2(xdataset::Measurement::String("world"));
    Value result = OperationMatrix({v1, v2});
    ASSERT_TRUE(result.is_meas());
    auto vec = result.as_meas().as_vector<std::string>();
    EXPECT_EQ(vec(0), "hello");
    EXPECT_EQ(vec(1), "world");
}

TEST(OperationMatrixTest, SameUnit)
{
    Unit u = Unit::parse("V");
    Value result = OperationMatrix({Value(xdataset::Measurement(1.0, u)),
                                     Value(xdataset::Measurement(2.0, u))});
    ASSERT_TRUE(result.is_meas());
    EXPECT_TRUE(result.as_meas().unit().same_dimension(u));
}

TEST(OperationMatrixTest, IncompatibleUnitsThrows)
{
    Unit uv = Unit::parse("V");
    Unit ua = Unit::parse("A");
    EXPECT_THROW(OperationMatrix({Value(xdataset::Measurement(1.0, uv)),
                                   Value(xdataset::Measurement(2.0, ua))}),
                 std::invalid_argument);
}

TEST(OperationMatrixTest, EmptyThrows)
{
    EXPECT_THROW(OperationMatrix({}), std::invalid_argument);
}

TEST(OperationMatrixTest, DataArraysSameKindSameShape)
{
    auto ds1 = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    Value v1(xdataset::DataArray::CreateIndependent(std::move(ds1)));
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = OperationMatrix({v1, v2});
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.size(), 2u);
}

TEST(OperationMatrixTest, PreservesFirstDataArrayMetadata)
{
    Block block(MakeBaseCreateInfo());
    DataArray da_z = block.GetOrCreateDataArray("z");
    Value v1(std::move(da_z));
    auto ds2 = xdataset::DataSeries::CreateScalarFromVector<double>({3.0});
    Value v2(xdataset::DataArray::CreateIndependent(std::move(ds2)));
    Value result = OperationMatrix({v1, v2});
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array();
    EXPECT_EQ(arr.multi_dimension_spec().rank(), 2u);
    EXPECT_EQ(arr.data_kind(), DataArrayKind::kDependent);
}

// =========================================================================
//  OperationSweep
// =========================================================================

TEST(OperationSweepTest, ScalarOnly)
{
    Value v1(xdataset::Measurement(1.0));
    Value v2(xdataset::Measurement(2.0));
    Value v3(xdataset::Measurement(3.0));
    Value result = OperationSweep({v1, v2, v3});
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.size(), 3u);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 1.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 2.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(2), 3.0);
}

TEST(OperationSweepTest, VectorOnly)
{
    VecXd a(2); a << 1.0, 2.0;
    VecXd b(2); b << 3.0, 4.0;
    Value result = OperationSweep({Value(xdataset::Measurement::Vector(a)),
                                    Value(xdataset::Measurement::Vector(b))});
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_EQ(arr.size(), 2u);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(0), 1.0);
    EXPECT_DOUBLE_EQ(arr.vector_at<double>(0)(1), 2.0);
}

TEST(OperationSweepTest, IntAndRealPromote)
{
    Value result = OperationSweep({Value(xdataset::Measurement(1)),
                                    Value(xdataset::Measurement(2.5))});
    ASSERT_TRUE(result.is_array());
    const auto& arr = result.as_array().data();
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(0), 1.0);
    EXPECT_DOUBLE_EQ(arr.scalar_at<double>(1), 2.5);
}

TEST(OperationSweepTest, EmptyThrows)
{
    EXPECT_THROW(OperationSweep({}), std::invalid_argument);
}

TEST(OperationSweepTest, IncompatibleUnitsThrows)
{
    Unit uv = Unit::parse("V");
    Unit ua = Unit::parse("A");
    EXPECT_THROW(OperationSweep({Value(xdataset::Measurement(1.0, uv)),
                                  Value(xdataset::Measurement(2.0, ua))}),
                 std::invalid_argument);
}
