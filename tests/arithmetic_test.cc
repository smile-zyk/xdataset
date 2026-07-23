#include "data_series.h"
#include "data_array.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstring>
#include <string>
#include <vector>


using xdataset::DataKind;
using xdataset::DataSeries;
using xdataset::Measurement;
using xdataset::DataType;
using xdataset::Index;
using xdataset::Unit;
using xdataset::DataArray;
using xdataset::DataArrayKind;
using xdataset::MultiDimensionSpec;
using xdataset::DimensionSpec;

// ---------------------------------------------------------------------------

// =========================================================================
//  DataSeries canonicalize (precondition for arithmetic)
// =========================================================================

TEST(DataSeriesUnitTest, CanonicalizeFastPathCoherentSI)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{9.0, 10.0});
    s.set_unit("Hz");  // multiplier == 1, non-affine
    s.canonicalized();
    // values unchanged
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(0), 9.0);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(1), 10.0);
    EXPECT_DOUBLE_EQ(s.unit().multiplier(), 1.0);
}

// =========================================================================
//  P3 Arithmetic: promoted_dtype
// =========================================================================

TEST(PromotedDTypeTest, IntegerPlusIntegerGivesInteger)
{
    EXPECT_EQ(xdataset::promoted_dtype(DataType::kInteger, DataType::kInteger), DataType::kInteger);
}

TEST(PromotedDTypeTest, IntegerPlusRealGivesReal)
{
    EXPECT_EQ(xdataset::promoted_dtype(DataType::kInteger, DataType::kReal), DataType::kReal);
    EXPECT_EQ(xdataset::promoted_dtype(DataType::kReal, DataType::kInteger), DataType::kReal);
}

TEST(PromotedDTypeTest, RealPlusRealGivesReal)
{
    EXPECT_EQ(xdataset::promoted_dtype(DataType::kReal, DataType::kReal), DataType::kReal);
}

TEST(PromotedDTypeTest, ComplexDominatesAll)
{
    EXPECT_EQ(xdataset::promoted_dtype(DataType::kReal, DataType::kComplex), DataType::kComplex);
    EXPECT_EQ(xdataset::promoted_dtype(DataType::kInteger, DataType::kComplex), DataType::kComplex);
    EXPECT_EQ(xdataset::promoted_dtype(DataType::kComplex, DataType::kComplex), DataType::kComplex);
}

TEST(PromotedDTypeTest, StringThrows)
{
    EXPECT_THROW(xdataset::promoted_dtype(DataType::kString, DataType::kReal), std::invalid_argument);
    EXPECT_THROW(xdataset::promoted_dtype(DataType::kReal, DataType::kString), std::invalid_argument);
}

// =========================================================================
//  P3 Arithmetic: DataSeries + DataSeries
// =========================================================================

TEST(DataSeriesArithTest, AddRealSeries)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    auto b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0, 6.0});
    DataSeries r = a + b;
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 7.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 9.0);
}

TEST(DataSeriesArithTest, AddIntegerSeries)
{
    auto a = DataSeries::CreateScalarFromVector<int>({1, 2, 3});
    auto b = DataSeries::CreateScalarFromVector<int>({10, 20, 30});
    DataSeries r = a + b;
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 11);
    EXPECT_EQ(r.scalar_at<int>(1), 22);
    EXPECT_EQ(r.scalar_at<int>(2), 33);
}

TEST(DataSeriesArithTest, AddIntegerToRealPromotes)
{
    auto a = DataSeries::CreateScalarFromVector<int>({1, 2});
    auto b = DataSeries::CreateScalarFromVector<double>({3.5, 4.5});
    DataSeries r = a + b;
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.5);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 6.5);
}

TEST(DataSeriesArithTest, AddComplexSeries)
{
    using cd = std::complex<double>;
    auto a = DataSeries::CreateScalarFromVector<cd>({cd(1, 2), cd(3, 4)});
    auto b = DataSeries::CreateScalarFromVector<cd>({cd(5, 6), cd(7, 8)});
    DataSeries r = a + b;
    EXPECT_EQ(r.data_type(), DataType::kComplex);
    EXPECT_DOUBLE_EQ(r.scalar_at<cd>(0).real(), 6.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<cd>(0).imag(), 8.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<cd>(1).real(), 10.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<cd>(1).imag(), 12.0);
}

TEST(DataSeriesArithTest, AddSizeMismatchThrows)
{
    auto a = DataSeries::CreateScalar<double>(3);
    auto b = DataSeries::CreateScalar<double>(2);
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataSeriesArithTest, AddStringSeriesThrows)
{
    auto a = DataSeries::CreateScalar<std::string>(2);
    auto b = DataSeries::CreateScalar<double>(2);
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataSeriesArithTest, AddUnitMismatchThrows)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    a.set_unit("meter");
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    b.set_unit(Unit::parse("sec"));
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataSeriesArithTest, AddWithOneDimensionless)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    // a is dimensionless
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    b.set_unit("meter");
    DataSeries r = a + b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 6.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(DataSeriesArithTest, AddWithUnitsDerivesResultUnit)
{
    auto a = DataSeries::CreateScalarFromVector<double>({100.0, 200.0});
    a.set_unit("cm");   // 1.0 m, 2.0 m after canonicalize
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    b.set_unit("meter");
    DataSeries r = a + b;
    // 1.0 + 3.0 = 4.0, 2.0 + 4.0 = 6.0
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 6.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(r.unit().multiplier(), 1.0);
}

// =========================================================================
//  P3 Arithmetic: DataSeries - DataSeries
// =========================================================================

TEST(DataSeriesArithTest, SubtractRealSeries)
{
    auto a = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 7.0});
    DataSeries r = a - b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 7.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 13.0);
}

TEST(DataSeriesArithTest, SubtractIntegerSeries)
{
    auto a = DataSeries::CreateScalarFromVector<int>({10, 20});
    auto b = DataSeries::CreateScalarFromVector<int>({3, 7});
    DataSeries r = a - b;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 7);
    EXPECT_EQ(r.scalar_at<int>(1), 13);
}

TEST(DataSeriesArithTest, SubtractUnitMismatchThrows)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0});
    a.set_unit(Unit::parse("V"));
    auto b = DataSeries::CreateScalarFromVector<double>({1.0});
    b.set_unit("Hz");
    EXPECT_THROW(a - b, std::invalid_argument);
}

TEST(DataSeriesArithTest, SubtractWithOneDimensionless)
{
    auto a = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    a.set_unit(Unit::parse("V"));
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 5.0});
    // b is dimensionless
    DataSeries r = a - b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 7.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 15.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("V")));
}

// =========================================================================
//  P3 Arithmetic: DataSeries * DataSeries
// =========================================================================

TEST(DataSeriesArithTest, MultiplyRealSeries)
{
    auto a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0, 4.0});
    auto b = DataSeries::CreateScalarFromVector<double>({5.0, 6.0, 7.0});
    DataSeries r = a * b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 10.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 18.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 28.0);
}

TEST(DataSeriesArithTest, MultiplyIntegerSeries)
{
    auto a = DataSeries::CreateScalarFromVector<int>({2, 3});
    auto b = DataSeries::CreateScalarFromVector<int>({4, 5});
    DataSeries r = a * b;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 8);
    EXPECT_EQ(r.scalar_at<int>(1), 15);
}

TEST(DataSeriesArithTest, MultiplyDerivesResultUnit)
{
    auto a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    a.set_unit("meter");
    auto b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0});
    b.set_unit(Unit::parse("sec"));
    DataSeries r = a * b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 8.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 15.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()*(
                               Unit::parse("sec").canonicalized())));
}

TEST(DataSeriesArithTest, MultiplyAmpereVoltYieldsWatt)
{
    auto a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    a.set_unit("A");
    auto b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0});
    b.set_unit("V");
    DataSeries r = a * b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 8.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 15.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("W")));
}

// =========================================================================
//  P3 Arithmetic: DataSeries / DataSeries
// =========================================================================

TEST(DataSeriesArithTest, DivideRealSeries)
{
    auto a = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    auto b = DataSeries::CreateScalarFromVector<double>({2.0, 4.0});
    DataSeries r = a / b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 5.0);
}

TEST(DataSeriesArithTest, DivideIntegerSeriesYieldsReal)
{
    auto a = DataSeries::CreateScalarFromVector<int>({10, 3});
    auto b = DataSeries::CreateScalarFromVector<int>({4, 2});
    DataSeries r = a / b;
    // Integer / Integer -> Real
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.5);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 1.5);
}

TEST(DataSeriesArithTest, DivideDerivesResultUnit)
{
    auto a = DataSeries::CreateScalarFromVector<double>({10.0});
    a.set_unit("meter");
    auto b = DataSeries::CreateScalarFromVector<double>({2.0});
    b.set_unit(Unit::parse("sec"));
    DataSeries r = a / b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()/(Unit::parse("sec").canonicalized())));
}

TEST(DataSeriesArithTest, DivideVoltAmpereYieldsOhm)
{
    auto a = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    a.set_unit("V");
    auto b = DataSeries::CreateScalarFromVector<double>({2.0, 4.0});
    b.set_unit("A");
    DataSeries r = a / b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 5.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("Ohm")));
}

// =========================================================================
//  P3 Arithmetic: DataSeries + Measurement (broadcast)
// =========================================================================

TEST(DataSeriesMeasArithTest, AddMeasurementReal)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Measurement m(10.0);
    DataSeries r = s + m;
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 11.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 12.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 13.0);
}

TEST(DataSeriesMeasArithTest, AddMeasurementInteger)
{
    auto s = DataSeries::CreateScalarFromVector<int>({1, 2, 3});
    Measurement m(100);
    DataSeries r = s + m;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 101);
    EXPECT_EQ(r.scalar_at<int>(1), 102);
    EXPECT_EQ(r.scalar_at<int>(2), 103);
}

TEST(DataSeriesMeasArithTest, AddMeasurementIntegerToRealPromotes)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.5, 2.5});
    Measurement m(3);
    DataSeries r = s + m;
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.5);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 5.5);
}

TEST(DataSeriesMeasArithTest, AddMeasurementWithUnits)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    s.set_unit("meter");
    Measurement m(100.0, xdataset::Unit::parse("cm"));  // 1.0 m after canonicalize
    DataSeries r = s + m;
    // 1.0 + 1.0 = 2.0, 2.0 + 1.0 = 3.0
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 3.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(DataSeriesMeasArithTest, AddMeasurementUnitMismatchThrows)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0});
    s.set_unit("meter");
    Measurement m(1.0, xdataset::Unit::parse("sec"));
    EXPECT_THROW(s + m, std::invalid_argument);
}

TEST(DataSeriesMeasArithTest, AddMeasurementOneDimensionless)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    s.set_unit("meter");
    Measurement m(100.0);  // dimensionless
    DataSeries r = s + m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 101.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 102.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(DataSeriesMeasArithTest, SubtractMeasurementOneDimensionless)
{
    auto s = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    s.set_unit("meter");
    Measurement m(3.0);  // dimensionless
    DataSeries r = s - m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 7.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 17.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(DataSeriesMeasArithTest, AddMeasurementStringThrows)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0});
    Measurement m(std::string("hello"));
    EXPECT_THROW(s + m, std::invalid_argument);
}

TEST(DataSeriesMeasArithTest, AddScalarDSToVectorMeasurement)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    Eigen::VectorXd v(3); v << 10.0, 20.0, 30.0;
    Measurement m(v);
    DataSeries r = s + m;
    // Scalar DS + Vector Meas -> Vector DataSeries
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    EXPECT_EQ(r.size(), 2u);
    auto row0 = r.vector_at<double>(0);
    auto row1 = r.vector_at<double>(1);
    EXPECT_DOUBLE_EQ(row0(0), 11.0); EXPECT_DOUBLE_EQ(row0(1), 21.0); EXPECT_DOUBLE_EQ(row0(2), 31.0);
    EXPECT_DOUBLE_EQ(row1(0), 12.0); EXPECT_DOUBLE_EQ(row1(1), 22.0); EXPECT_DOUBLE_EQ(row1(2), 32.0);
}

// =========================================================================
//  P3 Arithmetic: DataSeries - Measurement (broadcast)
// =========================================================================

TEST(DataSeriesMeasArithTest, SubtractMeasurement)
{
    auto s = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    Measurement m(3.0);
    DataSeries r = s - m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 7.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 17.0);
}

TEST(DataSeriesMeasArithTest, SubtractMeasurementWithUnits)
{
    auto s = DataSeries::CreateScalarFromVector<double>({5.0});
    s.set_unit("meter");
    Measurement m(200.0, xdataset::Unit::parse("cm"));  // 2.0 m
    DataSeries r = s - m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 3.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
}

// =========================================================================
//  P3 Arithmetic: DataSeries * Measurement (broadcast)
// =========================================================================

TEST(DataSeriesMeasArithTest, MultiplyMeasurement)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0, 3.0, 4.0});
    Measurement m(10.0);
    DataSeries r = s * m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 20.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 30.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 40.0);
}

TEST(DataSeriesMeasArithTest, MultiplyMeasurementDerivesResultUnit)
{
    auto s = DataSeries::CreateScalarFromVector<double>({3.0});
    s.set_unit("meter");
    Measurement m(2.0, xdataset::Unit::parse("sec"));
    DataSeries r = s * m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 6.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()*(Unit::parse("sec").canonicalized())));
}

TEST(DataSeriesMeasArithTest, MultiplyAmpereVoltMeasurementYieldsWatt)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    s.set_unit("A");
    Measurement m(4.0, xdataset::Unit::parse("V"));
    DataSeries r = s * m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 8.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 12.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("W")));
}

TEST(DataSeriesMeasArithTest, MultiplyMeasurementInteger)
{
    auto s = DataSeries::CreateScalarFromVector<int>({2, 3});
    Measurement m(5);
    DataSeries r = s * m;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 10);
    EXPECT_EQ(r.scalar_at<int>(1), 15);
}

// =========================================================================
//  P3 Arithmetic: DataSeries / Measurement (broadcast)
// =========================================================================

TEST(DataSeriesMeasArithTest, DivideMeasurement)
{
    auto s = DataSeries::CreateScalarFromVector<double>({10.0, 20.0, 30.0});
    Measurement m(2.0);
    DataSeries r = s / m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 10.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 15.0);
}

TEST(DataSeriesMeasArithTest, DivideMeasurementIntegerYieldsReal)
{
    auto s = DataSeries::CreateScalarFromVector<int>({10, 20});
    Measurement m(4);
    DataSeries r = s / m;
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.5);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 5.0);
}

TEST(DataSeriesMeasArithTest, DivideMeasurementDerivesResultUnit)
{
    auto s = DataSeries::CreateScalarFromVector<double>({10.0});
    s.set_unit("meter");
    Measurement m(2.0, xdataset::Unit::parse("sec"));
    DataSeries r = s / m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()/(Unit::parse("sec").canonicalized())));
}

TEST(DataSeriesMeasArithTest, DivideVoltAmpereMeasurementYieldsOhm)
{
    auto s = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    s.set_unit("V");
    Measurement m(2.0, xdataset::Unit::parse("A"));
    DataSeries r = s / m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 10.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("Ohm")));
}

// =========================================================================
//  P3: Dimensionless arithmetic
// =========================================================================

TEST(DimensionlessArithTest, AddDimensionlessSeries)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    // Both dimensionless -> same_dimension passes.
    DataSeries r = a + b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 6.0);
}

TEST(DimensionlessArithTest, MultiplyDimensionlessYieldsDimensionless)
{
    auto a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    auto b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0});
    DataSeries r = a * b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 8.0);
    EXPECT_TRUE(!r.unit().has_dimension());
}

TEST(DimensionlessArithTest, PowDimensionlessYieldsDimensionless)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0});
    DataSeries r = xdataset::pow(s, Measurement(3));
    EXPECT_TRUE(!r.unit().has_dimension());
}

// =========================================================================
//  Measurement: promoted_kind
// =========================================================================

TEST(PromotedKindTest, ScalarScalarGivesScalar)
{
    EXPECT_EQ(xdataset::promoted_kind(DataKind::kScalar, DataKind::kScalar), DataKind::kScalar);
}

TEST(PromotedKindTest, ScalarVectorGivesVector)
{
    EXPECT_EQ(xdataset::promoted_kind(DataKind::kScalar, DataKind::kVector), DataKind::kVector);
    EXPECT_EQ(xdataset::promoted_kind(DataKind::kVector, DataKind::kScalar), DataKind::kVector);
}

TEST(PromotedKindTest, ScalarMatrixGivesMatrix)
{
    EXPECT_EQ(xdataset::promoted_kind(DataKind::kScalar, DataKind::kMatrix), DataKind::kMatrix);
    EXPECT_EQ(xdataset::promoted_kind(DataKind::kMatrix, DataKind::kScalar), DataKind::kMatrix);
}

TEST(PromotedKindTest, VectorMatrixThrows)
{
    EXPECT_THROW(xdataset::promoted_kind(DataKind::kVector, DataKind::kMatrix), std::invalid_argument);
    EXPECT_THROW(xdataset::promoted_kind(DataKind::kMatrix, DataKind::kVector), std::invalid_argument);
}

// =========================================================================
//  Measurement + Measurement (scalar)
// =========================================================================

TEST(MeasArithTest, AddScalarReal)
{
    Measurement a(3.0), b(4.0);
    Measurement r = a + b;
    EXPECT_EQ(r.data_kind(), DataKind::kScalar);
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 7.0);
}

TEST(MeasArithTest, AddScalarInt)
{
    Measurement a(3), b(4);
    Measurement r = a + b;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.as_scalar<int>(), 7);
}

TEST(MeasArithTest, AddMixedIntRealPromotes)
{
    Measurement a(3.5), b(2);  // real + int -> real
    Measurement r = a + b;
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 5.5);
}

TEST(MeasArithTest, AddWithUnit)
{
    Measurement a(100.0, xdataset::Unit::parse("cm"));
    Measurement b(1.0, xdataset::Unit::parse("meter"));
    Measurement r = a + b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 2.0);  // 1.0 + 1.0
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(MeasArithTest, AddUnitMismatchThrows)
{
    Measurement a(1.0, xdataset::Unit::parse("meter"));
    Measurement b(1.0, xdataset::Unit::parse("sec"));
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(MeasArithTest, AddWithLeftDimensionless)
{
    Measurement a(2.0);                                   // dimensionless
    Measurement b(3.0, xdataset::Unit::parse("meter"));
    Measurement r = a + b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 5.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(MeasArithTest, AddWithRightDimensionless)
{
    Measurement a(5.0, xdataset::Unit::parse("meter"));
    Measurement b(3.0);                                   // dimensionless
    Measurement r = a + b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 8.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(MeasArithTest, SubtractWithLeftDimensionless)
{
    Measurement a(10.0);                                   // dimensionless
    Measurement b(3.0, xdataset::Unit::parse("sec"));
    Measurement r = a - b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 7.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("sec")));
}

TEST(MeasArithTest, SubtractWithRightDimensionless)
{
    Measurement a(10.0, xdataset::Unit::parse("Hz"));
    Measurement b(2.0);                                    // dimensionless
    Measurement r = a - b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 8.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("Hz")));
}

TEST(MeasArithTest, AddStringThrows)
{
    Measurement a(std::string("x"));
    Measurement b(1.0);
    EXPECT_THROW(a + b, std::invalid_argument);
}

// =========================================================================
//  Measurement + Measurement (broadcast)
// =========================================================================

TEST(MeasArithTest, AddScalarToVector)
{
    Eigen::VectorXd v(3); v << 1.0, 2.0, 3.0;
    Measurement vec(v);
    Measurement s(10.0);
    Measurement r = s + vec;
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 11.0);
    EXPECT_DOUBLE_EQ(rv(1), 12.0);
    EXPECT_DOUBLE_EQ(rv(2), 13.0);
}

TEST(MeasArithTest, AddScalarToMatrix)
{
    Eigen::MatrixXd m(2, 2); m << 1.0, 2.0, 3.0, 4.0;
    Measurement mat(m);
    Measurement s(10.0);
    Measurement r = s + mat;
    EXPECT_EQ(r.data_kind(), DataKind::kMatrix);
    auto rm = r.as_matrix<double>();
    EXPECT_DOUBLE_EQ(rm(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(rm(1, 1), 14.0);
}

TEST(MeasArithTest, AddVectorToVector)
{
    Eigen::VectorXd a(2); a << 1.0, 2.0;
    Eigen::VectorXd b(2); b << 3.0, 4.0;
    Measurement r = Measurement(a) + Measurement(b);
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 4.0);
    EXPECT_DOUBLE_EQ(rv(1), 6.0);
}

// =========================================================================
//  Measurement - Measurement
// =========================================================================

TEST(MeasArithTest, SubtractScalar)
{
    Measurement a(10.0), b(3.0);
    Measurement r = a - b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 7.0);
}

TEST(MeasArithTest, SubtractScalarFromVector)
{
    Eigen::VectorXd v(3); v << 10.0, 20.0, 30.0;
    Measurement vec(v), s(5.0);
    Measurement r = vec - s;
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 5.0);
    EXPECT_DOUBLE_EQ(rv(2), 25.0);
}

// =========================================================================
//  Measurement * Measurement
// =========================================================================

TEST(MeasArithTest, MultiplyScalar)
{
    Measurement a(6.0), b(7.0);
    Measurement r = a * b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 42.0);
}

TEST(MeasArithTest, MultiplyDerivesUnit)
{
    Measurement a(3.0, xdataset::Unit::parse("meter"));
    Measurement b(2.0, xdataset::Unit::parse("sec"));
    Measurement r = a * b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 6.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()*(Unit::parse("sec").canonicalized())));
}

TEST(MeasArithTest, MultiplyAmpereVoltYieldsWatt)
{
    Measurement a(2.0, xdataset::Unit::parse("A"));
    Measurement b(3.0, xdataset::Unit::parse("V"));
    Measurement r = a * b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 6.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("W")));
}

TEST(MeasArithTest, MultiplyVoltAmpereYieldsWatt)
{
    Measurement a(3.0, xdataset::Unit::parse("V"));
    Measurement b(2.0, xdataset::Unit::parse("A"));
    Measurement r = a * b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 6.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("W")));
}

TEST(MeasArithTest, MultiplyScalarToVector)
{
    Eigen::VectorXd v(2); v << 2.0, 3.0;
    Measurement vec(v), s(10.0);
    Measurement r = s * vec;
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 20.0);
    EXPECT_DOUBLE_EQ(rv(1), 30.0);
}

TEST(MeasArithTest, MultiplyIntScalars)
{
    Measurement a(3), b(5);
    Measurement r = a * b;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.as_scalar<int>(), 15);
}

// =========================================================================
//  Measurement / Measurement
// =========================================================================

TEST(MeasArithTest, DivideScalar)
{
    Measurement a(10.0), b(2.0);
    Measurement r = a / b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 5.0);
}

TEST(MeasArithTest, DivideIntYieldsReal)
{
    Measurement a(10), b(3);
    Measurement r = a / b;
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 10.0 / 3.0);
}

TEST(MeasArithTest, DivideDerivesUnit)
{
    Measurement a(10.0, xdataset::Unit::parse("meter"));
    Measurement b(2.0, xdataset::Unit::parse("sec"));
    Measurement r = a / b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 5.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()/(Unit::parse("sec").canonicalized())));
}

TEST(MeasArithTest, DivideVoltAmpereYieldsOhm)
{
    Measurement a(10.0, xdataset::Unit::parse("V"));
    Measurement b(2.0, xdataset::Unit::parse("A"));
    Measurement r = a / b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 5.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("Ohm")));
}

TEST(MeasArithTest, DivideWattAmpereYieldsVolt)
{
    Measurement a(6.0, xdataset::Unit::parse("W"));
    Measurement b(2.0, xdataset::Unit::parse("A"));
    Measurement r = a / b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 3.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("V")));
}

TEST(MeasArithTest, DivideWattVoltYieldsAmpere)
{
    Measurement a(6.0, xdataset::Unit::parse("W"));
    Measurement b(3.0, xdataset::Unit::parse("V"));
    Measurement r = a / b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 2.0);
    EXPECT_TRUE(r.unit().same_dimension(xdataset::Unit::parse("A")));
}

// =========================================================================
//  pow(Measurement, Measurement) -> scalar exponent (int values)
// =========================================================================

TEST(MeasPowTest, PowRealPositive)
{
    Measurement m(3.0);
    Measurement r = xdataset::pow(m, Measurement(2));
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 9.0);
}

TEST(MeasPowTest, PowRealZero)
{
    Measurement m(5.0);
    Measurement r = xdataset::pow(m, Measurement(0));
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 1.0);
}

TEST(MeasPowTest, PowRealNegative)
{
    Measurement m(2.0);
    Measurement r = xdataset::pow(m, Measurement(-1));
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 0.5);
}

TEST(MeasPowTest, PowVector)
{
    Eigen::VectorXd v(3); v << 2.0, 3.0, 4.0;
    Measurement r = xdataset::pow(Measurement(v), Measurement(2));
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 4.0);
    EXPECT_DOUBLE_EQ(rv(1), 9.0);
    EXPECT_DOUBLE_EQ(rv(2), 16.0);
}

TEST(MeasPowTest, PowMatrix)
{
    Eigen::MatrixXd m(2, 2); m << 2.0, 3.0, 4.0, 5.0;
    Measurement r = xdataset::pow(Measurement(m), Measurement(2));
    EXPECT_EQ(r.data_kind(), DataKind::kMatrix);
    auto rm = r.as_matrix<double>();
    EXPECT_DOUBLE_EQ(rm(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(rm(1, 1), 25.0);
}

TEST(MeasPowTest, PowDerivesUnit)
{
    Measurement m(3.0, xdataset::Unit::parse("meter"));
    Measurement r = xdataset::pow(m, Measurement(2));
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 9.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()));
}

TEST(MeasPowTest, PowStringThrows)
{
    Measurement m(std::string("x"));
    EXPECT_THROW(xdataset::pow(m, Measurement(2)), std::invalid_argument);
}

// =========================================================================
//  pow(Measurement, Measurement) -> broadcast
// =========================================================================

TEST(MeasPowMeasTest, PowMeasScalarExponent)
{
    Measurement base(3.0);
    Measurement exp(2);
    Measurement r = xdataset::pow(base, exp);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 9.0);
}

TEST(MeasPowMeasTest, PowMeasRealExponent)
{
    Measurement base(4.0);
    Measurement exp(0.5);
    Measurement r = xdataset::pow(base, exp);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 2.0);
}

TEST(MeasPowMeasTest, PowMeasExponentMustBeDimensionless)
{
    Measurement base(2.0);
    Measurement exp(3, xdataset::Unit::parse("meter"));
    EXPECT_THROW(xdataset::pow(base, exp), std::invalid_argument);
}

TEST(MeasPowMeasTest, PowMeasExponentMustNotBeString)
{
    Measurement base(2.0);
    Measurement exp(std::string("x"));
    EXPECT_THROW(xdataset::pow(base, exp), std::invalid_argument);
}

TEST(MeasPowMeasTest, PowMeasBroadcastScalarToVector)
{
    Measurement base(2.0);
    Eigen::VectorXd ev(3); ev << 1.0, 2.0, 3.0;
    Measurement exp(ev);
    Measurement r = xdataset::pow(base, exp);
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 2.0);
    EXPECT_DOUBLE_EQ(rv(1), 4.0);
    EXPECT_DOUBLE_EQ(rv(2), 8.0);
}

TEST(MeasPowMeasTest, PowMeasBroadcastVectorToScalar)
{
    Eigen::VectorXd bv(3); bv << 2.0, 3.0, 4.0;
    Measurement base(bv);
    Measurement exp(2);
    Measurement r = xdataset::pow(base, exp);
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 4.0);
    EXPECT_DOUBLE_EQ(rv(1), 9.0);
    EXPECT_DOUBLE_EQ(rv(2), 16.0);
}

TEST(MeasPowMeasTest, PowMeasWithUnitAndScalarExponent)
{
    Measurement base(2.0, xdataset::Unit::parse("meter"));
    Measurement exp(3);  // scalar int exponent -> unit derivation works
    Measurement r = xdataset::pow(base, exp);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 8.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()));
}

// =========================================================================
//  pow(DataSeries, Measurement)
// =========================================================================

TEST(DataSeriesPowMeasTest, PowMeasScalarExponent)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0, 3.0, 4.0});
    Measurement exp(2);
    DataSeries r = xdataset::pow(s, exp);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 9.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 16.0);
}

TEST(DataSeriesPowMeasTest, PowMeasRealExponent)
{
    auto s = DataSeries::CreateScalarFromVector<double>({4.0, 9.0});
    Measurement exp(0.5);
    DataSeries r = xdataset::pow(s, exp);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 3.0);
}

TEST(DataSeriesPowMeasTest, PowMeasWithUnits)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    s.set_unit("meter");
    Measurement exp(2);
    DataSeries r = xdataset::pow(s, exp);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 9.0);
    EXPECT_TRUE(r.unit().same_dimension(
        Unit::parse("meter").canonicalized()));
}

TEST(DataSeriesPowMeasTest, PowMeasExponentMustBeDimless)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0});
    Measurement exp(2, xdataset::Unit::parse("meter"));
    EXPECT_THROW(xdataset::pow(s, exp), std::invalid_argument);
}

TEST(DataSeriesPowMeasTest, PowMeasExponentNotString)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0});
    Measurement exp(std::string("x"));
    EXPECT_THROW(xdataset::pow(s, exp), std::invalid_argument);
}

// =========================================================================
//  Vector/Matrix DataSeries arithmetic (DataKind broadcast)
// =========================================================================

TEST(DataSeriesArithTest, ScalarDSPlusVectorDSYieldsVector)
{
    auto scalars = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    auto vectors = DataSeries::CreateVectorFromVector<double>(3,
        std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    DataSeries r = scalars + vectors;
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    EXPECT_EQ(r.size(), 2u);
    ASSERT_EQ(r.data_shape().size(), 1u);
    EXPECT_EQ(r.data_shape()[0], 3);
    auto row0 = r.vector_at<double>(0);
    auto row1 = r.vector_at<double>(1);
    EXPECT_DOUBLE_EQ(row0(0), 11.0); EXPECT_DOUBLE_EQ(row0(1), 12.0); EXPECT_DOUBLE_EQ(row0(2), 13.0);
    EXPECT_DOUBLE_EQ(row1(0), 24.0); EXPECT_DOUBLE_EQ(row1(1), 25.0); EXPECT_DOUBLE_EQ(row1(2), 26.0);
}

TEST(DataSeriesArithTest, ScalarDSTimesVectorDSYieldsVector)
{
    auto scalars = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    auto vectors = DataSeries::CreateVectorFromVector<double>(3,
        std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    DataSeries r = scalars * vectors;
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto row0 = r.vector_at<double>(0);
    auto row1 = r.vector_at<double>(1);
    EXPECT_DOUBLE_EQ(row0(0), 2.0);  EXPECT_DOUBLE_EQ(row0(1), 4.0);  EXPECT_DOUBLE_EQ(row0(2), 6.0);
    EXPECT_DOUBLE_EQ(row1(0), 12.0); EXPECT_DOUBLE_EQ(row1(1), 15.0); EXPECT_DOUBLE_EQ(row1(2), 18.0);
}

TEST(DataSeriesArithTest, VectorDSPlusVectorDS)
{
    auto a = DataSeries::CreateVectorFromVector<double>(3,
        std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto b = DataSeries::CreateVectorFromVector<double>(3,
        std::vector<double>{10.0, 20.0, 30.0, 40.0, 50.0, 60.0});
    DataSeries r = a + b;
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    EXPECT_EQ(r.size(), 2u);
    auto row0 = r.vector_at<double>(0);
    auto row1 = r.vector_at<double>(1);
    EXPECT_DOUBLE_EQ(row0(0), 11.0); EXPECT_DOUBLE_EQ(row0(2), 33.0);
    EXPECT_DOUBLE_EQ(row1(1), 55.0);
}

TEST(DataSeriesArithTest, VectorDSPlusScalarMeasurementYieldsVector)
{
    auto vecs = DataSeries::CreateVectorFromVector<double>(2,
        std::vector<double>{1.0, 2.0, 3.0, 4.0});
    Measurement s(10.0);
    DataSeries r = vecs + s;
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto row0 = r.vector_at<double>(0);
    auto row1 = r.vector_at<double>(1);
    EXPECT_DOUBLE_EQ(row0(0), 11.0); EXPECT_DOUBLE_EQ(row0(1), 12.0);
    EXPECT_DOUBLE_EQ(row1(0), 13.0); EXPECT_DOUBLE_EQ(row1(1), 14.0);
}

TEST(DataSeriesArithTest, ScalarDSPlusMatrixMeasurementYieldsMatrix)
{
    auto scalars = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    Eigen::MatrixXd m(2, 2); m << 1.0, 2.0, 3.0, 4.0;
    Measurement mat_meas(m);
    DataSeries r = scalars + mat_meas;
    EXPECT_EQ(r.data_kind(), DataKind::kMatrix);
    EXPECT_EQ(r.size(), 2u);
    auto row0 = r.matrix_at<double>(0);
    EXPECT_DOUBLE_EQ(row0(0, 0), 2.0); EXPECT_DOUBLE_EQ(row0(1, 1), 5.0);
    auto row1 = r.matrix_at<double>(1);
    EXPECT_DOUBLE_EQ(row1(0, 0), 3.0); EXPECT_DOUBLE_EQ(row1(1, 1), 6.0);
}

TEST(DataSeriesArithTest, VectorDSSizeMismatchThrows)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    auto b = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataSeriesArithTest, VectorDSShapeMismatchThrows)
{
    auto a = DataSeries::CreateVectorFromVector<double>(3,
        std::vector<double>{1.0, 2.0, 3.0});
    auto b = DataSeries::CreateVectorFromVector<double>(2,
        std::vector<double>{1.0, 2.0});
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataSeriesArithTest, VectorDSTimesMatrixDSThrows)
{
    auto vecs = DataSeries::CreateVectorFromVector<double>(3,
        std::vector<double>{1.0, 2.0, 3.0});
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {2, 2});
    mats.resize(1);
    mats.matrix_at<double>(0) << 1, 2, 3, 4;
    EXPECT_THROW(vecs + mats, std::invalid_argument);
}

// =========================================================================
//  pow(DataSeries, Measurement) with non-Scalar base
// =========================================================================

TEST(DataSeriesPowMeasTest, PowVectorDSWithScalarExponent)
{
    auto vecs = DataSeries::CreateVectorFromVector<double>(2,
        std::vector<double>{2.0, 3.0, 4.0, 5.0});
    Measurement exp(2);
    DataSeries r = xdataset::pow(vecs, exp);
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto row0 = r.vector_at<double>(0);
    EXPECT_DOUBLE_EQ(row0(0), 4.0);  // 2^2
    EXPECT_DOUBLE_EQ(row0(1), 9.0);  // 3^2
}

// =========================================================================
//  DataArray indep_names
// =========================================================================

TEST(DataArrayMetaTest, IndepNamesReturnsKeys)
{
    auto indep = DataArray::CreateIndependent("x",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0}));
    auto names = indep.indep_names();
    ASSERT_EQ(names.size(), 1u);
    EXPECT_EQ(names[0], "x");
}

// =========================================================================
//  DataArray + DataArray
// =========================================================================

TEST(DataArrayArithTest, AddScalarIndependentArrays)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    ds_a.set_unit(xdataset::Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0, 6.0});
    ds_b.set_unit(xdataset::Unit::parse("meter"));
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    auto r = a + b;
    EXPECT_EQ(r.data().data_kind(), DataKind::kScalar);
    EXPECT_EQ(r.data().size(), 3u);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), 9.0);
    EXPECT_TRUE(r.data().unit().same_dimension(xdataset::Unit::parse("meter")));
    // Result inherits structure from left operand
    EXPECT_EQ(r.multi_dimension_spec().rank(), a.multi_dimension_spec().rank());
    EXPECT_EQ(r.data_kind(), a.data_kind());
}

TEST(DataArrayArithTest, AddArrayUnitMismatchThrows)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({1.0});
    ds_a.set_unit(xdataset::Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({1.0});
    ds_b.set_unit(xdataset::Unit::parse("sec"));
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataArrayArithTest, AddArrayOneDimensionless)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    ds_a.set_unit(xdataset::Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    // b is dimensionless
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    auto r = a + b;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 6.0);
    EXPECT_TRUE(r.data().unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(DataArrayArithTest, AddArraySpecMismatchThrows)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0}));

    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0}));

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataArrayArithTest, AddArrayIndepMismatchThrows)
{
    // Create an independent array (1 indep = self)
    auto a = DataArray::CreateIndependent("x",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0}));

    // Create a dependent array with 2 independents
    auto indep1 = DataArray::CreateIndependent("p",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0}));
    auto indep2 = DataArray::CreateIndependent("q",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0}));
    auto b = DataArray::CreateDependent("y",
        DataSeries::CreateScalarFromVector<double>({10.0, 20.0, 30.0, 40.0}),
        {&indep1, &indep2});

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataArrayArithTest, SubtractArrays)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({10.0, 20.0}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<double>({3.0, 7.0}));

    auto r = a - b;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 7.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 13.0);
}

// =========================================================================
//  DataSeries pow  vs  Measurement (non-Scalar exponent)
// =========================================================================

TEST(DataSeriesPowTest, PowMeasurementVectorExponent)
{
    // base: scalar series {2.0, 3.0}, dimensionless
    auto base = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    // exponent: Vector([1.0, 2.0, 3.0])
    Measurement exp(Eigen::VectorXd{{1.0, 2.0, 3.0}});
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    EXPECT_EQ(r.size(), 2u);
    EXPECT_EQ(r.data_shape()[0], 3);
    EXPECT_DOUBLE_EQ(r.vector_at<double>(0)(0), std::pow(2.0, 1.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(0)(1), std::pow(2.0, 2.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(0)(2), std::pow(2.0, 3.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(1)(0), std::pow(3.0, 1.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(1)(2), std::pow(3.0, 3.0));
}

TEST(DataSeriesPowTest, PowMeasurementMatrixExponent)
{
    auto base = DataSeries::CreateScalarFromVector<double>({2.0});
    Eigen::MatrixXd m(2, 2); m << 1.0, 2.0, 3.0, 4.0;
    Measurement exp(m);
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_EQ(r.data_kind(), DataKind::kMatrix);
    EXPECT_EQ(r.size(), 1u);
    EXPECT_DOUBLE_EQ(r.matrix_at<double>(0)(0, 0), std::pow(2.0, 1.0));
    EXPECT_DOUBLE_EQ(r.matrix_at<double>(0)(1, 1), std::pow(2.0, 4.0));
}

TEST(DataSeriesPowTest, PowMeasurementWithUnitPreservesBaseUnit)
{
    auto base = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    base.set_unit(Unit::parse("meter"));
    Measurement exp(2);  // dimensionless scalar integer
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_TRUE(r.unit().same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 9.0);
}

TEST(DataSeriesPowTest, PowMeasurementDimensionedExponentThrows)
{
    auto base = DataSeries::CreateScalarFromVector<double>({2.0});
    Measurement exp(2.0, Unit::parse("meter"));
    EXPECT_THROW(xdataset::pow(base, exp), std::invalid_argument);
}

// =========================================================================
//  DataSeries pow  vs  DataSeries
// =========================================================================

TEST(DataSeriesPowTest, PowSeriesRowByRow)
{
    auto base = DataSeries::CreateScalarFromVector<double>({2.0, 3.0, 4.0});
    auto exp  = DataSeries::CreateScalarFromVector<double>({2.0, 3.0, 0.5});
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_EQ(r.size(), 3u);
    EXPECT_EQ(r.data_kind(), DataKind::kScalar);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), std::pow(2.0, 2.0));   // 4
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), std::pow(3.0, 3.0));   // 27
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), std::pow(4.0, 0.5));   // 2
}

TEST(DataSeriesPowTest, PowSeriesWithUnit)
{
    auto base = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    base.set_unit(Unit::parse("meter"));
    auto exp = DataSeries::CreateScalarFromVector<int>({2, 2});
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_TRUE(r.unit().same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 9.0);
}

TEST(DataSeriesPowTest, PowSeriesSizeMismatchThrows)
{
    auto base = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    auto exp  = DataSeries::CreateScalarFromVector<double>({2.0});
    EXPECT_THROW(xdataset::pow(base, exp), std::invalid_argument);
}

TEST(DataSeriesPowTest, PowSeriesDimensionalExponentThrows)
{
    auto base = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    auto exp  = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    exp.set_unit(Unit::parse("meter"));
    EXPECT_THROW(xdataset::pow(base, exp), std::invalid_argument);
}

TEST(DataSeriesPowTest, PowSeriesVectorBroadcast)
{
    // base: scalar series, exp: vector series -> result is vector
    auto base = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    auto exp  = DataSeries::CreateVector<double>(3, 2);
    exp.vector_at<double>(0) << 1.0, 2.0, 3.0;
    exp.vector_at<double>(1) << 0.5, 1.0, 1.5;
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    EXPECT_EQ(r.size(), 2u);
    EXPECT_DOUBLE_EQ(r.vector_at<double>(0)(0), std::pow(2.0, 1.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(0)(1), std::pow(2.0, 2.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(0)(2), std::pow(2.0, 3.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(1)(1), std::pow(3.0, 1.0));
}

// =========================================================================
//  pow(Measurement, DataSeries)
// =========================================================================

TEST(DataSeriesPowTest, PowMeasBaseSeriesExponent)
{
    Measurement base(2.0);
    auto exp = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_EQ(r.size(), 3u);
    EXPECT_EQ(r.data_kind(), DataKind::kScalar);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.0);   // 2^1
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 4.0);   // 2^2
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 8.0);   // 2^3
}

TEST(DataSeriesPowTest, PowMeasBaseWithUnit)
{
    Measurement base(2.0, Unit::parse("meter"));
    auto exp = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_TRUE(r.unit().same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 8.0);
}

TEST(DataSeriesPowTest, PowMeasBaseVectorExponent)
{
    Measurement base(2.0);
    auto exp = DataSeries::CreateVector<double>(2, 2);
    exp.vector_at<double>(0) << 1.0, 2.0;
    exp.vector_at<double>(1) << 0.5, 3.0;
    DataSeries r = xdataset::pow(base, exp);
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    EXPECT_EQ(r.size(), 2u);
    EXPECT_DOUBLE_EQ(r.vector_at<double>(0)(0), std::pow(2.0, 1.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(0)(1), std::pow(2.0, 2.0));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(1)(0), std::pow(2.0, 0.5));
    EXPECT_DOUBLE_EQ(r.vector_at<double>(1)(1), std::pow(2.0, 3.0));
}

TEST(DataSeriesPowTest, PowMeasBaseDimensionalExponentThrows)
{
    Measurement base(2.0);
    auto exp = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    exp.set_unit(Unit::parse("meter"));
    EXPECT_THROW(xdataset::pow(base, exp), std::invalid_argument);
}

// =========================================================================
//  DataArray pow  vs  DataArray
// =========================================================================

TEST(DataArrayPowTest, PowArrayRowByRow)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0, 3.0}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<double>({2.0, 3.0}));
    auto r = xdataset::pow(a, b);
    EXPECT_EQ(r.data().size(), 2u);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), std::pow(2.0, 2.0));
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), std::pow(3.0, 3.0));
}

TEST(DataArrayPowTest, PowArrayWithUnit)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    ds_a.set_unit(Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<int>({2, 2}));
    auto r = xdataset::pow(a, b);
    EXPECT_TRUE(r.data().unit().same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 9.0);
}

// =========================================================================
//  DataArray pow  vs  Measurement (non-Scalar exponent)
// =========================================================================

TEST(DataArrayPowTest, PowMeasurementNonScalar)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0, 3.0}));
    Measurement exp(Eigen::VectorXd{{1.0, 2.0, 3.0}});
    auto r = xdataset::pow(a, exp);
    EXPECT_EQ(r.data().data_kind(), DataKind::kVector);
    EXPECT_EQ(r.data().size(), 2u);
    EXPECT_DOUBLE_EQ(r.data().vector_at<double>(0)(0), std::pow(2.0, 1.0));
    EXPECT_DOUBLE_EQ(r.data().vector_at<double>(0)(1), std::pow(2.0, 2.0));
}

// =========================================================================
//  pow(Measurement, DataArray)
// =========================================================================

TEST(DataArrayPowTest, PowMeasBaseArrayExponent)
{
    Measurement base(2.0);
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0}));
    auto r = xdataset::pow(base, a);
    EXPECT_EQ(r.data().size(), 3u);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 2.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 4.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), 8.0);
    EXPECT_EQ(r.data_kind(), a.data_kind());
    EXPECT_EQ(r.multi_dimension_spec().rank(), a.multi_dimension_spec().rank());
}

TEST(DataArrayPowTest, PowMeasBaseArrayWithUnit)
{
    Measurement base(2.0, Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0, 3.0}));
    auto r = xdataset::pow(base, a);
    EXPECT_TRUE(r.data().unit().same_dimension(Unit::parse("meter")));
}

// =========================================================================
//  Measurement op DataSeries (reverse broadcast)
// =========================================================================

TEST(MeasDataSeriesArithTest, AddMeasurementToSeries)
{
    Measurement m(10.0);
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    DataSeries r = m + s;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 11.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 12.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 13.0);
}

TEST(MeasDataSeriesArithTest, SubtractMeasurementMinusSeries)
{
    Measurement m(10.0);
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    DataSeries r = m - s;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 9.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 8.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 7.0);
}

TEST(MeasDataSeriesArithTest, MultiplyMeasurementTimesSeries)
{
    Measurement m(2.0);
    auto s = DataSeries::CreateScalarFromVector<double>({3.0, 4.0, 5.0});
    DataSeries r = m * s;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 6.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 10.0);
}

TEST(MeasDataSeriesArithTest, DivideMeasurementBySeries)
{
    Measurement m(10.0);
    auto s = DataSeries::CreateScalarFromVector<double>({2.0, 5.0});
    DataSeries r = m / s;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 2.0);
}

TEST(MeasDataSeriesArithTest, AddMeasurementToSeriesWithUnits)
{
    Measurement m(100.0, Unit::parse("cm"));  // 1.0 m
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    s.set_unit(Unit::parse("meter"));
    DataSeries r = m + s;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 3.0);
    EXPECT_TRUE(r.unit().same_dimension(Unit::parse("meter")));
}

TEST(MeasDataSeriesArithTest, MeasurementTimesSeriesDerivesUnit)
{
    Measurement m(2.0, Unit::parse("A"));
    auto s = DataSeries::CreateScalarFromVector<double>({3.0});
    s.set_unit(Unit::parse("V"));
    DataSeries r = m * s;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 6.0);
    EXPECT_TRUE(r.unit().same_dimension(Unit::parse("W")));
}

// =========================================================================
//  Measurement op DataArray (reverse broadcast)
// =========================================================================

TEST(MeasDataArrayArithTest, AddMeasurementToArray)
{
    Measurement m(10.0);
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0}));
    auto r = m + a;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 11.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), 13.0);
    EXPECT_EQ(r.data_kind(), a.data_kind());
}

TEST(MeasDataArrayArithTest, SubtractMeasurementMinusArray)
{
    Measurement m(10.0);
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0}));
    auto r = m - a;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 9.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 8.0);
}

TEST(MeasDataArrayArithTest, MultiplyMeasurementTimesArray)
{
    Measurement m(2.0);
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({3.0, 4.0}));
    auto r = m * a;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 6.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 8.0);
}

TEST(MeasDataArrayArithTest, DivideMeasurementByArray)
{
    Measurement m(12.0);
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({3.0, 4.0}));
    auto r = m / a;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 3.0);
}

TEST(DataArrayArithTest, MultiplyArrays)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    ds_a.set_unit(xdataset::Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0});
    ds_b.set_unit(xdataset::Unit::parse("meter"));
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    auto r = a * b;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 8.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 15.0);
    EXPECT_TRUE(r.data().unit().same_dimension(
        Unit::parse("meter").canonicalized()*(
            Unit::parse("meter").canonicalized())));
}

TEST(DataArrayArithTest, DivideArrays)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    ds_a.set_unit(xdataset::Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({2.0, 4.0});
    ds_b.set_unit(xdataset::Unit::parse("sec"));
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    auto r = a / b;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 5.0);
    EXPECT_TRUE(r.data().unit().same_dimension(
        Unit::parse("meter").canonicalized()/(Unit::parse("sec").canonicalized())));
}

// =========================================================================
//  DataArray + Measurement
// =========================================================================

TEST(DataArrayMeasArithTest, AddMeasurement)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0}));
    Measurement m(10.0);

    auto r = a + m;
    EXPECT_EQ(r.data().size(), 3u);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 11.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), 13.0);
    EXPECT_EQ(r.data_kind(), a.data_kind());
}

TEST(DataArrayMeasArithTest, SubtractMeasurement)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({10.0, 20.0}));
    Measurement m(3.0);

    auto r = a - m;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 7.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 17.0);
}

TEST(DataArrayMeasArithTest, MultiplyMeasurement)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0, 3.0, 4.0}));
    Measurement m(10.0);

    auto r = a * m;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 20.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 30.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), 40.0);
}

TEST(DataArrayMeasArithTest, DivideMeasurement)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({10.0, 20.0, 30.0}));
    Measurement m(2.0);

    auto r = a / m;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), 15.0);
}

TEST(DataArrayMeasArithTest, AddMeasurementWithUnitMismatchThrows)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({1.0});
    ds_a.set_unit(xdataset::Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));
    Measurement m(1.0, xdataset::Unit::parse("sec"));

    EXPECT_THROW(a + m, std::invalid_argument);
}

TEST(DataArrayMeasArithTest, AddMeasurementOneDimensionless)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    ds_a.set_unit(xdataset::Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));
    Measurement m(100.0);  // dimensionless

    auto r = a + m;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 101.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 102.0);
    EXPECT_TRUE(r.data().unit().same_dimension(xdataset::Unit::parse("meter")));
}

// =========================================================================
//  pow(DataArray, Measurement)
// =========================================================================

TEST(DataArrayPowMeasTest, PowWithIntegerExponent)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0, 3.0, 4.0}));
    Measurement exp(2);

    auto r = xdataset::pow(a, exp);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 9.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), 16.0);
    EXPECT_EQ(r.multi_dimension_spec().rank(), a.multi_dimension_spec().rank());
}

TEST(DataArrayPowMeasTest, PowWithUnitDerivesUnit)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    ds_a.set_unit(xdataset::Unit::parse("meter"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));
    Measurement exp(2);

    auto r = xdataset::pow(a, exp);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 9.0);
    EXPECT_TRUE(r.data().unit().same_dimension(
        Unit::parse("meter").canonicalized()));
}

TEST(DataArrayPowMeasTest, PowExponentMustBeDimless)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0}));
    Measurement exp(2, xdataset::Unit::parse("meter"));
    EXPECT_THROW(xdataset::pow(a, exp), std::invalid_argument);
}

// =========================================================================
//  DataArray with Vector DataSeries (DataKind broadcast)
// =========================================================================

TEST(DataArrayArithTest, ScalarArrayPlusVectorArrayYieldsVector)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({10.0, 20.0}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateVectorFromVector<double>(3,
            std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));

    auto r = a + b;
    EXPECT_EQ(r.data().data_kind(), DataKind::kVector);
    EXPECT_EQ(r.data().size(), 2u);
    auto row0 = r.data().vector_at<double>(0);
    auto row1 = r.data().vector_at<double>(1);
    EXPECT_DOUBLE_EQ(row0(0), 11.0); EXPECT_DOUBLE_EQ(row0(2), 13.0);
    EXPECT_DOUBLE_EQ(row1(0), 24.0); EXPECT_DOUBLE_EQ(row1(2), 26.0);
}

// =========================================================================
//  DataArray result name
// =========================================================================

TEST(DataArrayArithTest, ResultHasAutoName)
{
    auto a = DataArray::CreateIndependent("temperature",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0}));
    auto b = DataArray::CreateIndependent("offset",
        DataSeries::CreateScalarFromVector<double>({10.0, 20.0}));

    auto r = a + b;
    EXPECT_EQ(r.name(), DataArray::kUnnamed);
}

// =========================================================================
//  Measurement comparison operators
// =========================================================================

TEST(MeasCmpTest, EqualScalar)
{
    Measurement r = Measurement(3.0) == Measurement(3.0);
    EXPECT_EQ(r.data_kind(), DataKind::kScalar);
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.as_scalar<int>(), 1);
}

TEST(MeasCmpTest, NotEqualScalar)
{
    Measurement r = Measurement(3.0) != Measurement(4.0);
    EXPECT_EQ(r.as_scalar<int>(), 1);
}

TEST(MeasCmpTest, LessThanScalar)
{
    Measurement r = Measurement(3.0) < Measurement(5.0);
    EXPECT_EQ(r.as_scalar<int>(), 1);
    r = Measurement(5.0) < Measurement(3.0);
    EXPECT_EQ(r.as_scalar<int>(), 0);
}

TEST(MeasCmpTest, GreaterThanScalar)
{
    Measurement r = Measurement(5.0) > Measurement(3.0);
    EXPECT_EQ(r.as_scalar<int>(), 1);
}

TEST(MeasCmpTest, LessEqualScalar)
{
    EXPECT_EQ((Measurement(3.0) <= Measurement(3.0)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(3.0) <= Measurement(5.0)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(5.0) <= Measurement(3.0)).as_scalar<int>(), 0);
}

TEST(MeasCmpTest, GreaterEqualScalar)
{
    EXPECT_EQ((Measurement(5.0) >= Measurement(3.0)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(3.0) >= Measurement(3.0)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(3.0) >= Measurement(5.0)).as_scalar<int>(), 0);
}

TEST(MeasCmpTest, CmpInteger)
{
    EXPECT_EQ((Measurement(5) == Measurement(5)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(5) != Measurement(3)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(3) < Measurement(5)).as_scalar<int>(), 1);
}

TEST(MeasCmpTest, CmpMixedIntReal)
{
    EXPECT_EQ((Measurement(3) == Measurement(3.0)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(3) < Measurement(3.5)).as_scalar<int>(), 1);
}

TEST(MeasCmpTest, CmpVector)
{
    Eigen::VectorXd a(3); a << 1.0, 2.0, 3.0;
    Eigen::VectorXd b(3); b << 1.0, 4.0, 3.0;
    Measurement r = Measurement(a) < Measurement(b);
    EXPECT_EQ(r.data_kind(), DataKind::kVector);
    auto rv = r.as_vector<int>();
    EXPECT_EQ(rv(0), 0);  // 1.0 < 1.0 false
    EXPECT_EQ(rv(1), 1);  // 2.0 < 4.0 true
    EXPECT_EQ(rv(2), 0);  // 3.0 < 3.0 false
}

TEST(MeasCmpTest, CmpUnitMismatchThrows)
{
    Measurement a(1.0, Unit::parse("meter"));
    Measurement b(1.0, Unit::parse("sec"));
    EXPECT_THROW(a == b, std::invalid_argument);
}

TEST(MeasCmpTest, CmpStringThrows)
{
    Measurement a(1.0);
    Measurement b(std::string("x"));
    EXPECT_THROW(a == b, std::invalid_argument);
}

// =========================================================================
//  Measurement logical operators
// =========================================================================

TEST(MeasLogicalTest, AndScalar)
{
    EXPECT_EQ((Measurement(1.0) && Measurement(1.0)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(1.0) && Measurement(0.0)).as_scalar<int>(), 0);
    EXPECT_EQ((Measurement(0.0) && Measurement(1.0)).as_scalar<int>(), 0);
}

TEST(MeasLogicalTest, OrScalar)
{
    EXPECT_EQ((Measurement(1.0) || Measurement(0.0)).as_scalar<int>(), 1);
    EXPECT_EQ((Measurement(0.0) || Measurement(0.0)).as_scalar<int>(), 0);
    EXPECT_EQ((Measurement(0) || Measurement(1)).as_scalar<int>(), 1);
}

TEST(MeasLogicalTest, AndVector)
{
    Eigen::VectorXd a(2); a << 1.0, 0.0;
    Eigen::VectorXd b(2); b << 1.0, 1.0;
    Measurement r = Measurement(a) && Measurement(b);
    auto rv = r.as_vector<int>();
    EXPECT_EQ(rv(0), 1);
    EXPECT_EQ(rv(1), 0);
}

// =========================================================================
//  Measurement bitwise operators
// =========================================================================

TEST(MeasBitwiseTest, And)
{
    Measurement r = Measurement(6) & Measurement(3);  // 110 & 011 = 010
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.as_scalar<int>(), 2);
}

TEST(MeasBitwiseTest, Or)
{
    Measurement r = Measurement(6) | Measurement(3);  // 110 | 011 = 111
    EXPECT_EQ(r.as_scalar<int>(), 7);
}

TEST(MeasBitwiseTest, Xor)
{
    Measurement r = Measurement(6) ^ Measurement(3);  // 110 ^ 011 = 101
    EXPECT_EQ(r.as_scalar<int>(), 5);
}

TEST(MeasBitwiseTest, BitwiseOnRealThrows)
{
    EXPECT_THROW(Measurement(1.0) & Measurement(1.0), std::invalid_argument);
}

// =========================================================================
//  Measurement shift operators
// =========================================================================

TEST(MeasShiftTest, LeftShift)
{
    Measurement r = Measurement(1) << Measurement(3);
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.as_scalar<int>(), 8);
}

TEST(MeasShiftTest, RightShift)
{
    Measurement r = Measurement(8) >> Measurement(2);
    EXPECT_EQ(r.as_scalar<int>(), 2);
}

TEST(MeasShiftTest, ShiftOnRealThrows)
{
    EXPECT_THROW(Measurement(1.0) << Measurement(1), std::invalid_argument);
}

// =========================================================================
//  Measurement modulo
// =========================================================================

TEST(MeasModTest, ModInteger)
{
    Measurement r = Measurement(10) % Measurement(3);
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.as_scalar<int>(), 1);
}

TEST(MeasModTest, ModReal)
{
    Measurement r = Measurement(10.5) % Measurement(3.0);
    EXPECT_EQ(r.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), std::fmod(10.5, 3.0));
}

TEST(MeasModTest, ModComplexThrows)
{
    EXPECT_THROW(Measurement(std::complex<double>(1, 2)) % Measurement(3.0), std::invalid_argument);
}

// =========================================================================
//  Measurement unary operators
// =========================================================================

TEST(MeasUnaryTest, NegateScalar)
{
    Measurement m(3.0);
    Measurement r = -m;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), -3.0);
}

TEST(MeasUnaryTest, NegatePreservesUnit)
{
    Measurement m(3.0, Unit::parse("meter"));
    Measurement r = -m;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), -3.0);
    EXPECT_TRUE(r.unit().same_dimension(Unit::parse("meter")));
}

TEST(MeasUnaryTest, NegateVector)
{
    Eigen::VectorXd v(2); v << 1.0, -2.0;
    Measurement r = -Measurement(v);
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), -1.0);
    EXPECT_DOUBLE_EQ(rv(1), 2.0);
}

TEST(MeasUnaryTest, LogicalNotScalar)
{
    EXPECT_EQ((!Measurement(0.0)).as_scalar<int>(), 1);
    EXPECT_EQ((!Measurement(1.0)).as_scalar<int>(), 0);
    EXPECT_EQ((!Measurement(0)).as_scalar<int>(), 1);
    EXPECT_EQ((!Measurement(5)).as_scalar<int>(), 0);
}

TEST(MeasUnaryTest, LogicalNotVector)
{
    Eigen::VectorXd v(3); v << 1.0, 0.0, 3.0;
    Measurement r = !Measurement(v);
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    auto rv = r.as_vector<int>();
    EXPECT_EQ(rv(0), 0);
    EXPECT_EQ(rv(1), 1);
    EXPECT_EQ(rv(2), 0);
}

TEST(MeasUnaryTest, BitwiseNot)
{
    Measurement r = ~Measurement(0);  // ~0 = -1 (all bits set in 2's complement)
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.as_scalar<int>(), ~0);
}

TEST(MeasUnaryTest, BitwiseNotOnRealThrows)
{
    EXPECT_THROW(~Measurement(1.0), std::invalid_argument);
}

// =========================================================================
//  DataSeries comparison operators
// =========================================================================

TEST(DataSeriesCmpTest, Equal)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    auto b = DataSeries::CreateScalarFromVector<double>({1.0, 0.0, 3.0});
    DataSeries r = a == b;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 1);
    EXPECT_EQ(r.scalar_at<int>(1), 0);
    EXPECT_EQ(r.scalar_at<int>(2), 1);
}

TEST(DataSeriesCmpTest, LessThan)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 5.0});
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 2.0});
    DataSeries r = a < b;
    EXPECT_EQ(r.scalar_at<int>(0), 1);
    EXPECT_EQ(r.scalar_at<int>(1), 0);
}

TEST(DataSeriesCmpTest, CmpSizeMismatchThrows)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    auto b = DataSeries::CreateScalarFromVector<double>({1.0});
    EXPECT_THROW(a == b, std::invalid_argument);
}

// =========================================================================
//  DataSeries logical operators
// =========================================================================

TEST(DataSeriesLogicalTest, And)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 0.0, 2.0});
    auto b = DataSeries::CreateScalarFromVector<double>({1.0, 1.0, 0.0});
    DataSeries r = a && b;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 1);
    EXPECT_EQ(r.scalar_at<int>(1), 0);
    EXPECT_EQ(r.scalar_at<int>(2), 0);
}

TEST(DataSeriesLogicalTest, Or)
{
    auto a = DataSeries::CreateScalarFromVector<double>({0.0, 0.0, 1.0});
    auto b = DataSeries::CreateScalarFromVector<double>({0.0, 1.0, 0.0});
    DataSeries r = a || b;
    EXPECT_EQ(r.scalar_at<int>(0), 0);
    EXPECT_EQ(r.scalar_at<int>(1), 1);
    EXPECT_EQ(r.scalar_at<int>(2), 1);
}

// =========================================================================
//  DataSeries bitwise / shift / modulo
// =========================================================================

TEST(DataSeriesBitwiseTest, And)
{
    auto a = DataSeries::CreateScalarFromVector<int>({6, 3});
    auto b = DataSeries::CreateScalarFromVector<int>({3, 5});
    DataSeries r = a & b;
    EXPECT_EQ(r.scalar_at<int>(0), 2);
    EXPECT_EQ(r.scalar_at<int>(1), 1);
}

TEST(DataSeriesShiftTest, LeftShift)
{
    auto a = DataSeries::CreateScalarFromVector<int>({1, 2});
    auto b = DataSeries::CreateScalarFromVector<int>({3, 1});
    DataSeries r = a << b;
    EXPECT_EQ(r.scalar_at<int>(0), 8);
    EXPECT_EQ(r.scalar_at<int>(1), 4);
}

TEST(DataSeriesModTest, Mod)
{
    auto a = DataSeries::CreateScalarFromVector<int>({10, 15});
    auto b = DataSeries::CreateScalarFromVector<int>({3, 4});
    DataSeries r = a % b;
    EXPECT_EQ(r.scalar_at<int>(0), 1);
    EXPECT_EQ(r.scalar_at<int>(1), 3);
}

// =========================================================================
//  DataSeries unary operators
// =========================================================================

TEST(DataSeriesUnaryTest, Negate)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, -2.0, 3.0});
    DataSeries r = -s;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), -1.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 2.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), -3.0);
}

TEST(DataSeriesUnaryTest, LogicalNot)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 0.0, 5.0});
    DataSeries r = !s;
    EXPECT_EQ(r.data_type(), DataType::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 0);
    EXPECT_EQ(r.scalar_at<int>(1), 1);
    EXPECT_EQ(r.scalar_at<int>(2), 0);
}

TEST(DataSeriesUnaryTest, BitwiseNot)
{
    auto s = DataSeries::CreateScalarFromVector<int>({0, -1, 1});
    DataSeries r = ~s;
    EXPECT_EQ(r.scalar_at<int>(0), ~0);
    EXPECT_EQ(r.scalar_at<int>(1), ~(-1));
    EXPECT_EQ(r.scalar_at<int>(2), ~1);
}

// =========================================================================
//  DataSeries op Measurement (new operators)
// =========================================================================

TEST(DataSeriesMeasCmpTest, EqualMeas)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0, 2.0, 3.0});
    Measurement m(2.0);
    DataSeries r = s == m;
    EXPECT_EQ(r.scalar_at<int>(0), 1);
    EXPECT_EQ(r.scalar_at<int>(1), 1);
    EXPECT_EQ(r.scalar_at<int>(2), 0);
}

TEST(DataSeriesMeasCmpTest, LessThanMeas)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 5.0});
    Measurement m(3.0);
    DataSeries r = s < m;
    EXPECT_EQ(r.scalar_at<int>(0), 1);
    EXPECT_EQ(r.scalar_at<int>(1), 0);
}

TEST(DataSeriesMeasLogicalTest, AndMeas)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 0.0});
    Measurement m(1.0);
    DataSeries r = s && m;
    EXPECT_EQ(r.scalar_at<int>(0), 1);
    EXPECT_EQ(r.scalar_at<int>(1), 0);
}

// =========================================================================
//  Measurement op DataSeries (new operators)
// =========================================================================

TEST(MeasDataSeriesCmpTest, Equal)
{
    Measurement m(2.0);
    auto s = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    DataSeries r = m == s;
    EXPECT_EQ(r.scalar_at<int>(0), 1);
    EXPECT_EQ(r.scalar_at<int>(1), 0);
}

TEST(MeasDataSeriesLogicalTest, Or)
{
    Measurement m(0.0);
    auto s = DataSeries::CreateScalarFromVector<double>({0.0, 1.0});
    DataSeries r = m || s;
    EXPECT_EQ(r.scalar_at<int>(0), 0);
    EXPECT_EQ(r.scalar_at<int>(1), 1);
}

TEST(MeasDataSeriesBitwiseTest, And)
{
    Measurement m(7);
    auto s = DataSeries::CreateScalarFromVector<int>({3, 5});
    DataSeries r = m & s;
    EXPECT_EQ(r.scalar_at<int>(0), 3);
    EXPECT_EQ(r.scalar_at<int>(1), 5);
}

// =========================================================================
//  DataArray comparison
// =========================================================================

TEST(DataArrayCmpTest, EqualScalar)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<double>({1.0, 0.0, 3.0}));

    auto r = a == b;
    EXPECT_EQ(r.data().data_type(), DataType::kInteger);
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 0);
    EXPECT_EQ(r.data().scalar_at<int>(2), 1);
}

TEST(DataArrayCmpTest, LessThan)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 5.0}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<double>({3.0, 2.0}));

    auto r = a < b;
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 0);
}

// =========================================================================
//  DataArray logical
// =========================================================================

TEST(DataArrayLogicalTest, And)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 0.0}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<double>({1.0, 1.0}));

    auto r = a && b;
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 0);
}

TEST(DataArrayLogicalTest, Or)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({0.0, 0.0}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<double>({1.0, 0.0}));

    auto r = a || b;
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 0);
}

// =========================================================================
//  DataArray bitwise / shift / modulo
// =========================================================================

TEST(DataArrayBitwiseTest, And)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<int>({6, 3}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<int>({3, 5}));

    auto r = a & b;
    EXPECT_EQ(r.data().scalar_at<int>(0), 2);
    EXPECT_EQ(r.data().scalar_at<int>(1), 1);
}

TEST(DataArrayShiftTest, LeftShift)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<int>({1, 2}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<int>({3, 1}));

    auto r = a << b;
    EXPECT_EQ(r.data().scalar_at<int>(0), 8);
    EXPECT_EQ(r.data().scalar_at<int>(1), 4);
}

TEST(DataArrayModTest, Mod)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<int>({10, 15}));
    auto b = DataArray::CreateIndependent("b",
        DataSeries::CreateScalarFromVector<int>({3, 4}));

    auto r = a % b;
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 3);
}

// =========================================================================
//  DataArray unary
// =========================================================================

TEST(DataArrayUnaryTest, Negate)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, -2.0, 3.0}));

    auto r = -a;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), -1.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 2.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), -3.0);
}

TEST(DataArrayUnaryTest, LogicalNot)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 0.0, 5.0}));

    auto r = !a;
    EXPECT_EQ(r.data().data_type(), DataType::kInteger);
    EXPECT_EQ(r.data().scalar_at<int>(0), 0);
    EXPECT_EQ(r.data().scalar_at<int>(1), 1);
    EXPECT_EQ(r.data().scalar_at<int>(2), 0);
}

// =========================================================================
//  DataArray op Measurement (new operators)
// =========================================================================

TEST(DataArrayMeasCmpTest, EqualMeas)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0, 2.0, 3.0}));
    Measurement m(2.0);

    auto r = a == m;
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 1);
    EXPECT_EQ(r.data().scalar_at<int>(2), 0);
}

TEST(DataArrayMeasCmpTest, LessThanMeas)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 5.0}));
    Measurement m(3.0);

    auto r = a < m;
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 0);
}

TEST(DataArrayMeasLogicalTest, AndMeas)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({1.0, 0.0}));
    Measurement m(1.0);

    auto r = a && m;
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 0);
}

// =========================================================================
//  Measurement op DataArray (new operators)
// =========================================================================

TEST(MeasDataArrayCmpTest, Equal)
{
    Measurement m(2.0);
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0, 3.0}));

    auto r = m == a;
    EXPECT_EQ(r.data().scalar_at<int>(0), 1);
    EXPECT_EQ(r.data().scalar_at<int>(1), 0);
}

TEST(MeasDataArrayLogicalTest, Or)
{
    Measurement m(0.0);
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({0.0, 1.0}));

    auto r = m || a;
    EXPECT_EQ(r.data().scalar_at<int>(0), 0);
    EXPECT_EQ(r.data().scalar_at<int>(1), 1);
    EXPECT_EQ(r.data_kind(), a.data_kind());
}
