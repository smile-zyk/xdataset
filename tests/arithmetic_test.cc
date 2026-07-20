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
using xdataset::DTypeTag;
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

TEST(DataSeriesUnitTest, CanonicalizeConvertsValues)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{1.0, 5.0});
    s.set_unit("km");
    s.canonicalize();
    // 1 km  1000 m,  5 km  5000 m
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(0), 1000.0);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(1), 5000.0);
    EXPECT_DOUBLE_EQ(s.unit().multiplier(), 1.0);
    EXPECT_TRUE(xdataset::same_dimension(s.unit(), xdataset::parse_unit("m")));
}

TEST(DataSeriesUnitTest, CanonicalizeFastPathCoherentSI)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{9.0, 10.0});
    s.set_unit("Hz");  // multiplier == 1, non-affine
    s.canonicalize();
    // values unchanged
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(0), 9.0);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(1), 10.0);
    EXPECT_DOUBLE_EQ(s.unit().multiplier(), 1.0);
}

TEST(DataSeriesUnitTest, CanonicalizeAffineDegC)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{0.0, 100.0});
    s.set_unit("degC");
    s.canonicalize();
    // 0 C  273.15 K,  100 C  373.15 K
    EXPECT_NEAR(s.scalar_at<double>(0), 273.15, 1e-10);
    EXPECT_NEAR(s.scalar_at<double>(1), 373.15, 1e-10);
    EXPECT_FALSE(xdataset::is_affine(s.unit()));
}

TEST(DataSeriesUnitTest, CanonicalizedDoesNotModifyOriginal)
{
    DataSeries orig = DataSeries::CreateScalarFromVector<double>(std::vector<double>{2.0, 4.0});
    orig.set_unit("mm");

    DataSeries copy = orig.canonicalized();
    // copy: 2 mm  0.002 m, 4 mm  0.004 m
    EXPECT_NEAR(copy.scalar_at<double>(0), 0.002, 1e-12);
    EXPECT_TRUE(xdataset::same_dimension(copy.unit(), xdataset::parse_unit("m")));

    // orig unchanged
    EXPECT_DOUBLE_EQ(orig.scalar_at<double>(0), 2.0);
    EXPECT_TRUE(xdataset::same_dimension(orig.unit(), xdataset::parse_unit("mm")));
}


// =========================================================================
//  P3 Arithmetic: promoted_dtype
// =========================================================================

TEST(PromotedDTypeTest, IntegerPlusIntegerGivesInteger)
{
    EXPECT_EQ(xdataset::promoted_dtype(DTypeTag::kInteger, DTypeTag::kInteger), DTypeTag::kInteger);
}

TEST(PromotedDTypeTest, IntegerPlusRealGivesReal)
{
    EXPECT_EQ(xdataset::promoted_dtype(DTypeTag::kInteger, DTypeTag::kReal), DTypeTag::kReal);
    EXPECT_EQ(xdataset::promoted_dtype(DTypeTag::kReal, DTypeTag::kInteger), DTypeTag::kReal);
}

TEST(PromotedDTypeTest, RealPlusRealGivesReal)
{
    EXPECT_EQ(xdataset::promoted_dtype(DTypeTag::kReal, DTypeTag::kReal), DTypeTag::kReal);
}

TEST(PromotedDTypeTest, ComplexDominatesAll)
{
    EXPECT_EQ(xdataset::promoted_dtype(DTypeTag::kReal, DTypeTag::kComplex), DTypeTag::kComplex);
    EXPECT_EQ(xdataset::promoted_dtype(DTypeTag::kInteger, DTypeTag::kComplex), DTypeTag::kComplex);
    EXPECT_EQ(xdataset::promoted_dtype(DTypeTag::kComplex, DTypeTag::kComplex), DTypeTag::kComplex);
}

TEST(PromotedDTypeTest, StringThrows)
{
    EXPECT_THROW(xdataset::promoted_dtype(DTypeTag::kString, DTypeTag::kReal), std::invalid_argument);
    EXPECT_THROW(xdataset::promoted_dtype(DTypeTag::kReal, DTypeTag::kString), std::invalid_argument);
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
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
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
    EXPECT_EQ(r.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 11);
    EXPECT_EQ(r.scalar_at<int>(1), 22);
    EXPECT_EQ(r.scalar_at<int>(2), 33);
}

TEST(DataSeriesArithTest, AddIntegerToRealPromotes)
{
    auto a = DataSeries::CreateScalarFromVector<int>({1, 2});
    auto b = DataSeries::CreateScalarFromVector<double>({3.5, 4.5});
    DataSeries r = a + b;
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.5);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 6.5);
}

TEST(DataSeriesArithTest, AddComplexSeries)
{
    using cd = std::complex<double>;
    auto a = DataSeries::CreateScalarFromVector<cd>({cd(1, 2), cd(3, 4)});
    auto b = DataSeries::CreateScalarFromVector<cd>({cd(5, 6), cd(7, 8)});
    DataSeries r = a + b;
    EXPECT_EQ(r.dtype(), DTypeTag::kComplex);
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
    a.set_unit("m");
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    b.set_unit("s");
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataSeriesArithTest, AddWithOneDimensionless)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    // a is dimensionless
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    b.set_unit("m");
    DataSeries r = a + b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 6.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("m")));
}

TEST(DataSeriesArithTest, AddWithUnitsDerivesResultUnit)
{
    auto a = DataSeries::CreateScalarFromVector<double>({100.0, 200.0});
    a.set_unit("cm");   // 1.0 m, 2.0 m after canonicalize
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    b.set_unit("m");
    DataSeries r = a + b;
    // 1.0 + 3.0 = 4.0, 2.0 + 4.0 = 6.0
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 6.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("m")));
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
    EXPECT_EQ(r.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 7);
    EXPECT_EQ(r.scalar_at<int>(1), 13);
}

TEST(DataSeriesArithTest, SubtractUnitMismatchThrows)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0});
    a.set_unit("Pa");
    auto b = DataSeries::CreateScalarFromVector<double>({1.0});
    b.set_unit("Hz");
    EXPECT_THROW(a - b, std::invalid_argument);
}

TEST(DataSeriesArithTest, SubtractWithOneDimensionless)
{
    auto a = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    a.set_unit("Pa");
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 5.0});
    // b is dimensionless
    DataSeries r = a - b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 7.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 15.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("Pa")));
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
    EXPECT_EQ(r.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 8);
    EXPECT_EQ(r.scalar_at<int>(1), 15);
}

TEST(DataSeriesArithTest, MultiplyDerivesResultUnit)
{
    auto a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    a.set_unit("m");
    auto b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0});
    b.set_unit("s");
    DataSeries r = a * b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 8.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 15.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::multiply_dim(xdataset::canonicalize(xdataset::parse_unit("m")),
                               xdataset::canonicalize(xdataset::parse_unit("s")))));
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
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.5);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 1.5);
}

TEST(DataSeriesArithTest, DivideDerivesResultUnit)
{
    auto a = DataSeries::CreateScalarFromVector<double>({10.0});
    a.set_unit("m");
    auto b = DataSeries::CreateScalarFromVector<double>({2.0});
    b.set_unit("s");
    DataSeries r = a / b;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::canonicalize(xdataset::parse_unit("m/s"))));
}

// =========================================================================
//  P3 Arithmetic: DataSeries + Measurement (broadcast)
// =========================================================================

TEST(DataSeriesMeasArithTest, AddMeasurementReal)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
    Measurement m(10.0);
    DataSeries r = s + m;
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 11.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 12.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(2), 13.0);
}

TEST(DataSeriesMeasArithTest, AddMeasurementInteger)
{
    auto s = DataSeries::CreateScalarFromVector<int>({1, 2, 3});
    Measurement m(100);
    DataSeries r = s + m;
    EXPECT_EQ(r.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(r.scalar_at<int>(0), 101);
    EXPECT_EQ(r.scalar_at<int>(1), 102);
    EXPECT_EQ(r.scalar_at<int>(2), 103);
}

TEST(DataSeriesMeasArithTest, AddMeasurementIntegerToRealPromotes)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.5, 2.5});
    Measurement m(3);
    DataSeries r = s + m;
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.5);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 5.5);
}

TEST(DataSeriesMeasArithTest, AddMeasurementWithUnits)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    s.set_unit("m");
    Measurement m(100.0, xdataset::parse_unit("cm"));  // 1.0 m after canonicalize
    DataSeries r = s + m;
    // 1.0 + 1.0 = 2.0, 2.0 + 1.0 = 3.0
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 3.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("m")));
}

TEST(DataSeriesMeasArithTest, AddMeasurementUnitMismatchThrows)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0});
    s.set_unit("m");
    Measurement m(1.0, xdataset::parse_unit("s"));
    EXPECT_THROW(s + m, std::invalid_argument);
}

TEST(DataSeriesMeasArithTest, AddMeasurementOneDimensionless)
{
    auto s = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    s.set_unit("m");
    Measurement m(100.0);  // dimensionless
    DataSeries r = s + m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 101.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 102.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("m")));
}

TEST(DataSeriesMeasArithTest, SubtractMeasurementOneDimensionless)
{
    auto s = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    s.set_unit("m");
    Measurement m(3.0);  // dimensionless
    DataSeries r = s - m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 7.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 17.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("m")));
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
    // Scalar DS + Vector Meas → Vector DataSeries
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
    s.set_unit("m");
    Measurement m(200.0, xdataset::parse_unit("cm"));  // 2.0 m
    DataSeries r = s - m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 3.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("m")));
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
    s.set_unit("m");
    Measurement m(2.0, xdataset::parse_unit("s"));
    DataSeries r = s * m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 6.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::canonicalize(xdataset::parse_unit("m*s"))));
}

TEST(DataSeriesMeasArithTest, MultiplyMeasurementInteger)
{
    auto s = DataSeries::CreateScalarFromVector<int>({2, 3});
    Measurement m(5);
    DataSeries r = s * m;
    EXPECT_EQ(r.dtype(), DTypeTag::kInteger);
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
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 2.5);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 5.0);
}

TEST(DataSeriesMeasArithTest, DivideMeasurementDerivesResultUnit)
{
    auto s = DataSeries::CreateScalarFromVector<double>({10.0});
    s.set_unit("m");
    Measurement m(2.0, xdataset::parse_unit("s"));
    DataSeries r = s / m;
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 5.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::canonicalize(xdataset::parse_unit("m/s"))));
}

// =========================================================================
//  P3: Dimensionless arithmetic
// =========================================================================

TEST(DimensionlessArithTest, AddDimensionlessSeries)
{
    auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    auto b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    // Both dimensionless — same_dimension passes.
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
    EXPECT_TRUE(xdataset::is_dimensionless(r.unit()));
}

TEST(DimensionlessArithTest, PowDimensionlessYieldsDimensionless)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0});
    DataSeries r = xdataset::pow(s, Measurement(3));
    EXPECT_TRUE(xdataset::is_dimensionless(r.unit()));
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
    EXPECT_EQ(r.kind(), DataKind::kScalar);
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 7.0);
}

TEST(MeasArithTest, AddScalarInt)
{
    Measurement a(3), b(4);
    Measurement r = a + b;
    EXPECT_EQ(r.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(r.as_scalar<int>(), 7);
}

TEST(MeasArithTest, AddMixedIntRealPromotes)
{
    Measurement a(3.5), b(2);  // real + int → real
    Measurement r = a + b;
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 5.5);
}

TEST(MeasArithTest, AddWithUnit)
{
    Measurement a(100.0, xdataset::parse_unit("cm"));
    Measurement b(1.0, xdataset::parse_unit("m"));
    Measurement r = a + b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 2.0);  // 1.0 + 1.0
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("m")));
}

TEST(MeasArithTest, AddUnitMismatchThrows)
{
    Measurement a(1.0, xdataset::parse_unit("m"));
    Measurement b(1.0, xdataset::parse_unit("s"));
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(MeasArithTest, AddWithLeftDimensionless)
{
    Measurement a(2.0);                                   // dimensionless
    Measurement b(3.0, xdataset::parse_unit("m"));
    Measurement r = a + b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 5.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("m")));
}

TEST(MeasArithTest, AddWithRightDimensionless)
{
    Measurement a(5.0, xdataset::parse_unit("kg"));
    Measurement b(3.0);                                   // dimensionless
    Measurement r = a + b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 8.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("kg")));
}

TEST(MeasArithTest, SubtractWithLeftDimensionless)
{
    Measurement a(10.0);                                   // dimensionless
    Measurement b(3.0, xdataset::parse_unit("s"));
    Measurement r = a - b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 7.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("s")));
}

TEST(MeasArithTest, SubtractWithRightDimensionless)
{
    Measurement a(10.0, xdataset::parse_unit("N"));
    Measurement b(2.0);                                    // dimensionless
    Measurement r = a - b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 8.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(), xdataset::parse_unit("N")));
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
    EXPECT_EQ(r.kind(), DataKind::kVector);
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
    EXPECT_EQ(r.kind(), DataKind::kMatrix);
    auto rm = r.as_matrix<double>();
    EXPECT_DOUBLE_EQ(rm(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(rm(1, 1), 14.0);
}

TEST(MeasArithTest, AddVectorToVector)
{
    Eigen::VectorXd a(2); a << 1.0, 2.0;
    Eigen::VectorXd b(2); b << 3.0, 4.0;
    Measurement r = Measurement(a) + Measurement(b);
    EXPECT_EQ(r.kind(), DataKind::kVector);
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
    Measurement a(3.0, xdataset::parse_unit("m"));
    Measurement b(2.0, xdataset::parse_unit("s"));
    Measurement r = a * b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 6.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::canonicalize(xdataset::parse_unit("m*s"))));
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
    EXPECT_EQ(r.dtype(), DTypeTag::kInteger);
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
    EXPECT_EQ(r.dtype(), DTypeTag::kReal);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 10.0 / 3.0);
}

TEST(MeasArithTest, DivideDerivesUnit)
{
    Measurement a(10.0, xdataset::parse_unit("m"));
    Measurement b(2.0, xdataset::parse_unit("s"));
    Measurement r = a / b;
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 5.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::canonicalize(xdataset::parse_unit("m/s"))));
}

// =========================================================================
//  pow(Measurement, Measurement) — scalar exponent (int values)
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
    EXPECT_EQ(r.kind(), DataKind::kVector);
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 4.0);
    EXPECT_DOUBLE_EQ(rv(1), 9.0);
    EXPECT_DOUBLE_EQ(rv(2), 16.0);
}

TEST(MeasPowTest, PowMatrix)
{
    Eigen::MatrixXd m(2, 2); m << 2.0, 3.0, 4.0, 5.0;
    Measurement r = xdataset::pow(Measurement(m), Measurement(2));
    EXPECT_EQ(r.kind(), DataKind::kMatrix);
    auto rm = r.as_matrix<double>();
    EXPECT_DOUBLE_EQ(rm(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(rm(1, 1), 25.0);
}

TEST(MeasPowTest, PowDerivesUnit)
{
    Measurement m(3.0, xdataset::parse_unit("m"));
    Measurement r = xdataset::pow(m, Measurement(2));
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 9.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::canonicalize(xdataset::parse_unit("m^2"))));
}

TEST(MeasPowTest, PowStringThrows)
{
    Measurement m(std::string("x"));
    EXPECT_THROW(xdataset::pow(m, Measurement(2)), std::invalid_argument);
}

// =========================================================================
//  pow(Measurement, Measurement) — broadcast
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
    Measurement exp(3, xdataset::parse_unit("m"));
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
    EXPECT_EQ(r.kind(), DataKind::kVector);
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
    EXPECT_EQ(r.kind(), DataKind::kVector);
    auto rv = r.as_vector<double>();
    EXPECT_DOUBLE_EQ(rv(0), 4.0);
    EXPECT_DOUBLE_EQ(rv(1), 9.0);
    EXPECT_DOUBLE_EQ(rv(2), 16.0);
}

TEST(MeasPowMeasTest, PowMeasNonScalarExpRequiresDimlessBase)
{
    // Base has unit "m", exponent is vector → must throw
    Measurement base(2.0, xdataset::parse_unit("m"));
    Eigen::VectorXd ev(2); ev << 1.0, 2.0;
    Measurement exp(ev);
    EXPECT_THROW(xdataset::pow(base, exp), std::invalid_argument);
}

TEST(MeasPowMeasTest, PowMeasWithUnitAndScalarExponent)
{
    Measurement base(2.0, xdataset::parse_unit("m"));
    Measurement exp(3);  // scalar int exponent → unit derivation works
    Measurement r = xdataset::pow(base, exp);
    EXPECT_DOUBLE_EQ(r.as_scalar<double>(), 8.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::canonicalize(xdataset::parse_unit("m^3"))));
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
    s.set_unit("m");
    Measurement exp(2);
    DataSeries r = xdataset::pow(s, exp);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.scalar_at<double>(1), 9.0);
    EXPECT_TRUE(xdataset::same_dimension(r.unit(),
        xdataset::canonicalize(xdataset::parse_unit("m^2"))));
}

TEST(DataSeriesPowMeasTest, PowMeasExponentMustBeDimless)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0});
    Measurement exp(2, xdataset::parse_unit("m"));
    EXPECT_THROW(xdataset::pow(s, exp), std::invalid_argument);
}

TEST(DataSeriesPowMeasTest, PowMeasExponentMustBeScalar)
{
    auto s = DataSeries::CreateScalarFromVector<double>({2.0});
    Eigen::VectorXd v(2); v << 1.0, 2.0;
    Measurement exp(v);
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
    DataSeries mats(DataKind::kMatrix, DTypeTag::kReal, {2, 2});
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
    ds_a.set_unit(xdataset::parse_unit("m"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0, 6.0});
    ds_b.set_unit(xdataset::parse_unit("m"));
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    auto r = a + b;
    EXPECT_EQ(r.data().data_kind(), DataKind::kScalar);
    EXPECT_EQ(r.data().size(), 3u);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(2), 9.0);
    EXPECT_TRUE(xdataset::same_dimension(r.data().unit(), xdataset::parse_unit("m")));
    // Result inherits structure from left operand
    EXPECT_EQ(r.multi_dimension_spec().rank(), a.multi_dimension_spec().rank());
    EXPECT_EQ(r.kind(), a.kind());
}

TEST(DataArrayArithTest, AddArrayUnitMismatchThrows)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({1.0});
    ds_a.set_unit(xdataset::parse_unit("m"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({1.0});
    ds_b.set_unit(xdataset::parse_unit("s"));
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(DataArrayArithTest, AddArrayOneDimensionless)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    ds_a.set_unit(xdataset::parse_unit("m"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({3.0, 4.0});
    // b is dimensionless
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    auto r = a + b;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 6.0);
    EXPECT_TRUE(xdataset::same_dimension(r.data().unit(), xdataset::parse_unit("m")));
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

TEST(DataArrayArithTest, MultiplyArrays)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
    ds_a.set_unit(xdataset::parse_unit("m"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({4.0, 5.0});
    ds_b.set_unit(xdataset::parse_unit("m"));
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    auto r = a * b;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 8.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 15.0);
    EXPECT_TRUE(xdataset::same_dimension(r.data().unit(),
        xdataset::canonicalize(xdataset::parse_unit("m^2"))));
}

TEST(DataArrayArithTest, DivideArrays)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({10.0, 20.0});
    ds_a.set_unit(xdataset::parse_unit("m"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));

    auto ds_b = DataSeries::CreateScalarFromVector<double>({2.0, 4.0});
    ds_b.set_unit(xdataset::parse_unit("s"));
    auto b = DataArray::CreateIndependent("b", std::move(ds_b));

    auto r = a / b;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 5.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 5.0);
    EXPECT_TRUE(xdataset::same_dimension(r.data().unit(),
        xdataset::canonicalize(xdataset::parse_unit("m/s"))));
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
    EXPECT_EQ(r.kind(), a.kind());
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
    ds_a.set_unit(xdataset::parse_unit("m"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));
    Measurement m(1.0, xdataset::parse_unit("s"));

    EXPECT_THROW(a + m, std::invalid_argument);
}

TEST(DataArrayMeasArithTest, AddMeasurementOneDimensionless)
{
    auto ds_a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0});
    ds_a.set_unit(xdataset::parse_unit("m"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));
    Measurement m(100.0);  // dimensionless

    auto r = a + m;
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 101.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 102.0);
    EXPECT_TRUE(xdataset::same_dimension(r.data().unit(), xdataset::parse_unit("m")));
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
    ds_a.set_unit(xdataset::parse_unit("m"));
    auto a = DataArray::CreateIndependent("a", std::move(ds_a));
    Measurement exp(2);

    auto r = xdataset::pow(a, exp);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(0), 4.0);
    EXPECT_DOUBLE_EQ(r.data().scalar_at<double>(1), 9.0);
    EXPECT_TRUE(xdataset::same_dimension(r.data().unit(),
        xdataset::canonicalize(xdataset::parse_unit("m^2"))));
}

TEST(DataArrayPowMeasTest, PowExponentMustBeDimless)
{
    auto a = DataArray::CreateIndependent("a",
        DataSeries::CreateScalarFromVector<double>({2.0}));
    Measurement exp(2, xdataset::parse_unit("m"));
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
