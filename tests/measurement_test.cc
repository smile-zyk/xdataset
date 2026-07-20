#include "data_series.h"

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

// ---------------------------------------------------------------------------

// =========================================================================
//  Measurement -- construction, type queries, unit
// =========================================================================

TEST(CellTest, ScalarCellCreateAndMutate) {
    Measurement m = Measurement(42.0);
    EXPECT_EQ(m.kind(), DataKind::kScalar);
    EXPECT_EQ(m.dtype(), DTypeTag::kReal);
    EXPECT_EQ(m.dtype_name(), "Real");
    EXPECT_DOUBLE_EQ(m.as_scalar<double>(), 42.0);

    m = Measurement(3.5);
    EXPECT_DOUBLE_EQ(m.as_scalar<double>(), 3.5);
}

TEST(CellTest, IntegerCellDtype) {
    Measurement m = Measurement(7);
    EXPECT_EQ(m.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(m.dtype_name(), "Integer");
    EXPECT_EQ(m.as_scalar<int>(), 7);
}

TEST(CellTest, ComplexCellDtype) {
    using cd = std::complex<double>;
    Measurement m = Measurement(cd(1.0, 2.0));
    EXPECT_EQ(m.dtype(), DTypeTag::kComplex);
    EXPECT_EQ(m.dtype_name(), "Complex");
    EXPECT_DOUBLE_EQ(m.as_scalar<cd>().real(), 1.0);
    EXPECT_DOUBLE_EQ(m.as_scalar<cd>().imag(), 2.0);
}

TEST(CellTest, StringCellDtype) {
    Measurement m = Measurement(std::string("hello"));
    EXPECT_EQ(m.dtype(), DTypeTag::kString);
    EXPECT_EQ(m.dtype_name(), "String");
    EXPECT_EQ(m.as_scalar<std::string>(), "hello");
}

TEST(CellTest, AppendCellToSeries) {
    Measurement m = Measurement(3.5);
    DataSeries s = DataSeries::CreateScalar<double>(0);
    s.append(Measurement(1.25));
    s.append(m);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(1), 3.5);
}

TEST(CellTest, AppendMismatchedCellThrows) {
    Measurement int_cell = Measurement(10);
    DataSeries s = DataSeries::CreateScalar<double>(0);
    EXPECT_THROW(s.append(int_cell), std::bad_cast);
}

TEST(CellTest, AppendUnitMismatchThrows) {
    // First append succeeds and sets the series unit.
    Measurement m_m = Measurement(1.0, xdataset::Unit::parse("meter"));
    DataSeries s = DataSeries::CreateScalar<double>(0);
    s.append(m_m);
    EXPECT_TRUE(s.unit().same_dimension(xdataset::Unit::parse("meter")));

    // Subsequent append with incompatible unit must throw.
    Measurement m_s = Measurement(2.0, xdataset::Unit::parse("sec"));
    EXPECT_THROW(s.append(m_s), std::invalid_argument);
}

TEST(CellTest, CellAtRoundtripScalar) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{1, 2, 3});
    Measurement m = s.measurement_at(1);
    EXPECT_EQ(m.kind(), DataKind::kScalar);
    EXPECT_EQ(m.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(m.as_scalar<int>(), 2);
}

TEST(CellTest, CellAtRoundtripVector) {
    DataSeries vecs(DataKind::kVector, DTypeTag::kReal, {3});
    vecs.resize(2);
    vecs.vector_at<double>(0) << 1.0, 2.0, 3.0;
    vecs.vector_at<double>(1) << 4.0, 5.0, 6.0;
    Measurement m = vecs.measurement_at(1);
    EXPECT_EQ(m.kind(), DataKind::kVector);
    EXPECT_DOUBLE_EQ(m.as_vector<double>()(0), 4.0);
    EXPECT_DOUBLE_EQ(m.as_vector<double>()(2), 6.0);
}

TEST(CellTest, CellAtRoundtripMatrix) {
    DataSeries mats(DataKind::kMatrix, DTypeTag::kReal, {2, 2});
    mats.resize(1);
    mats.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    Measurement m = mats.measurement_at(0);
    EXPECT_EQ(m.kind(), DataKind::kMatrix);
    EXPECT_DOUBLE_EQ(m.as_matrix<double>()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.as_matrix<double>()(1, 1), 4.0);
}

// ---------------------------------------------------------------------------
// Iterator  forward traversal
// ---------------------------------------------------------------------------

// Unit
// ---------------------------------------------------------------------------

TEST(CellUnitTest, DefaultCellIsDimensionless)
{
    Measurement m;
    EXPECT_TRUE(m.unit().same_dimension(xdataset::Unit()));
}

TEST(CellUnitTest, CopyPropagatesUnit)
{
    Measurement m = Measurement(3.14);
    m.set_unit(xdataset::Unit::parse("meter"));
    Measurement m2(m);
    EXPECT_TRUE(m2.unit().same_dimension(m.unit()));
}

TEST(CellUnitTest, MovePropagatesUnit)
{
    Measurement m = Measurement(2.72);
    m.set_unit(xdataset::Unit::parse("Hz"));
    Measurement m2(std::move(m));
    EXPECT_TRUE(m2.unit().same_dimension(xdataset::Unit::parse("Hz")));
}

TEST(CellUnitTest, AssignPropagatesUnit)
{
    Measurement m1 = Measurement(1.0);
    m1.set_unit(xdataset::Unit::parse("meter"));
    Measurement m2;
    m2 = m1;
    EXPECT_TRUE(m2.unit().same_dimension(xdataset::Unit::parse("meter")));
}

// ---------------------------------------------------------------------------
// Unit  DataSeries set_unit / canonicalize / canonicalized

// =========================================================================
//  Measurement: canonicalized
// =========================================================================

TEST(MeasurementCanonTest, CanonicalizedCmToMeter)
{
    Measurement m(5.0, xdataset::Unit::parse("cm"));
    Measurement c = m.canonicalized();
    // 5 cm 鈫?0.05 m
    EXPECT_DOUBLE_EQ(c.as_scalar<double>(), 0.05);
    EXPECT_DOUBLE_EQ(c.unit().multiplier(), 1.0);
    EXPECT_TRUE(c.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(MeasurementCanonTest, CanonicalizedFastPath)
{
    Measurement m(3.0, xdataset::Unit::parse("Hz"));  // already coherent SI
    Measurement c = m.canonicalized();
    EXPECT_DOUBLE_EQ(c.as_scalar<double>(), 3.0);
    EXPECT_TRUE(c.unit().same_dimension(xdataset::Unit::parse("Hz")));
}

TEST(MeasurementCanonTest, CanonicalizedStringNoValueChange)
{
    Measurement m(std::string("hello"), xdataset::Unit::parse("meter"));
    Measurement c = m.canonicalized();
    EXPECT_EQ(c.as_scalar<std::string>(), "hello");
    EXPECT_TRUE(c.unit().same_dimension(xdataset::Unit::parse("meter")));
}

// =========================================================================
//  MeasurementFormatter auto-scale
// =========================================================================

TEST(MeasurementFormatTest, AutoScaleMega)
{
    Measurement m(1e9, xdataset::Unit::parse("Hz"));
    std::string s = m.to_string();
    // 1e9 Hz 鈫?1 GHz
    EXPECT_TRUE(s.find("GHz") != std::string::npos);
}

TEST(MeasurementFormatTest, AutoScaleMilli)
{
    Measurement m(0.002, xdataset::Unit::parse("V"));
    std::string s = m.to_string();
    // 0.002 V 鈫?2 mV
    EXPECT_TRUE(s.find("2") != std::string::npos);
    EXPECT_TRUE(s.find("mV") != std::string::npos);
}

TEST(MeasurementFormatTest, AutoScaleKiloMeter)
{
    Measurement m(5000, xdataset::Unit::parse("meter"));
    std::string s = m.to_string();
    // 5000 meter 鈫?5 Kmeter
    EXPECT_TRUE(s.find("5") != std::string::npos);
    EXPECT_TRUE(s.find("Kmeter") != std::string::npos);
}

TEST(MeasurementFormatTest, AutoScaleMilliMeter)
{
    Measurement m(0.003, xdataset::Unit::parse("meter"));
    std::string s = m.to_string();
    // 0.003 meter 鈫?3 mmeter
    EXPECT_TRUE(s.find("3") != std::string::npos);
    EXPECT_TRUE(s.find("mmeter") != std::string::npos);
}

TEST(MeasurementFormatTest, AutoScaleNoneForDimensionless)
{
    Measurement m(3.14);
    std::string s = m.to_string();
    // 3.14 stays 3.14 (in [1, 1000))
    EXPECT_NE(s.find("3.14"), std::string::npos);
}

TEST(MeasurementFormatTest, AutoScaleMegaDimensionless)
{
    Measurement m(1000000);
    std::string s = m.to_string();
    // 1000000 鈫?1 M (dimensionless auto-scale)
    EXPECT_TRUE(s.find("M") != std::string::npos);
    EXPECT_TRUE(s.find("1") != std::string::npos);
}

TEST(MeasurementFormatTest, AutoScaleKiloFor100Hz)
{
    Measurement m(100.0, xdataset::Unit::parse("Hz"));
    std::string s = m.to_string();
    // 100 Hz stays 100 Hz (1 <= 100 < 1000)
    EXPECT_TRUE(s.find("100") != std::string::npos);
}

// =========================================================================
