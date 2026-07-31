// =============================================================================
//  xdataset -- value_test.cc
// =============================================================================
//
//  Tests for the Value class only.

#include "value.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using xdataset::DataKind;
using xdataset::DataType;
using xdataset::Index;
using xdataset::Unit;
using xdataset::VecXd;
using xdataset::VecXi;
using xdataset::VecXcd;
using xdataset::VecXs;
using xdataset::MatXd;
using xdataset::MatXi;
using xdataset::MatXcd;
using xdataset::MatXs;
using xdataset::Value;

// =========================================================================
//  Construction
// =========================================================================

TEST(ValueTest, DefaultIsMeasurement)
{
    Value v;
    EXPECT_TRUE(v.is_meas());
    EXPECT_FALSE(v.is_array());
}

TEST(ValueTest, ConstructFromMeasurement)
{
    xdataset::Measurement m(3.14, Unit::parse("meter"));
    Value v(m);
    EXPECT_TRUE(v.is_meas());
    EXPECT_DOUBLE_EQ(v.as_meas().as_scalar<double>(), 3.14);
}

TEST(ValueTest, ConstructFromMeasurementInPlace)
{
    Value v(xdataset::Measurement(42));
    EXPECT_TRUE(v.is_meas());
    EXPECT_EQ(v.as_meas().as_scalar<int>(), 42);
}

TEST(ValueTest, ConstructFromDataArray)
{
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    auto da = xdataset::DataArray::CreateIndependent(std::move(ds));
    Value v(da);
    EXPECT_TRUE(v.is_array());
    EXPECT_EQ(v.as_array().data().size(), 3u);
}

// =========================================================================
//  Metadata (unified access)
// =========================================================================

TEST(ValueTest, MetadataMeas)
{
    VecXd ev5(5); ev5.setOnes();
    xdataset::Measurement m(ev5, Unit::parse("Hz"));
    Value v(m);
    EXPECT_EQ(v.data_kind(), DataKind::kVector);
    EXPECT_EQ(v.data_type(), DataType::kReal);
    EXPECT_EQ(v.shape(), xdataset::DataShape{5});
    EXPECT_TRUE(v.unit().same_dimension(Unit::parse("Hz")));
    EXPECT_EQ(v.rows(), 1);
}

TEST(ValueTest, MetadataArray)
{
    auto ds = xdataset::DataSeries::CreateScalarFromVector<double>({10.0, 20.0, 30.0});
    ds.set_unit(Unit::parse("V"));
    auto da = xdataset::DataArray::CreateIndependent(std::move(ds));
    Value v(da);
    EXPECT_EQ(v.data_kind(), DataKind::kScalar);
    EXPECT_EQ(v.data_type(), DataType::kReal);
    EXPECT_TRUE(v.shape().empty());
    EXPECT_TRUE(v.unit().same_dimension(Unit::parse("V")));
    EXPECT_EQ(v.rows(), 3);
}
