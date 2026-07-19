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
    Measurement m_m = Measurement(1.0, xdataset::parse_unit("m"));
    DataSeries s = DataSeries::CreateScalar<double>(0);
    s.append(m_m);
    EXPECT_TRUE(xdataset::same_dimension(s.unit(), xdataset::parse_unit("m")));

    // Subsequent append with incompatible unit must throw.
    Measurement m_s = Measurement(2.0, xdataset::parse_unit("s"));
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
    EXPECT_TRUE(xdataset::same_dimension(m.unit(), xdataset::Unit()));
}

TEST(CellUnitTest, CopyPropagatesUnit)
{
    Measurement m = Measurement(3.14);
    m.set_unit(xdataset::parse_unit("m"));
    Measurement m2(m);
    EXPECT_TRUE(xdataset::same_dimension(m2.unit(), m.unit()));
}

TEST(CellUnitTest, MovePropagatesUnit)
{
    Measurement m = Measurement(2.72);
    m.set_unit(xdataset::parse_unit("Hz"));
    Measurement m2(std::move(m));
    EXPECT_TRUE(xdataset::same_dimension(m2.unit(), xdataset::parse_unit("Hz")));
}

TEST(CellUnitTest, AssignPropagatesUnit)
{
    Measurement m1 = Measurement(1.0);
    m1.set_unit(xdataset::parse_unit("m"));
    Measurement m2;
    m2 = m1;
    EXPECT_TRUE(xdataset::same_dimension(m2.unit(), xdataset::parse_unit("m")));
}

// ---------------------------------------------------------------------------
// Unit  DataSeries set_unit / canonicalize / canonicalized

// =========================================================================
//  Measurement: canonicalized
// =========================================================================

TEST(MeasurementCanonTest, CanonicalizedKmToM)
{
    Measurement m(5.0, xdataset::parse_unit("km"));
    Measurement c = m.canonicalized();
    EXPECT_DOUBLE_EQ(c.as_scalar<double>(), 5000.0);
    EXPECT_DOUBLE_EQ(c.unit().multiplier(), 1.0);
    EXPECT_TRUE(xdataset::same_dimension(c.unit(), xdataset::parse_unit("m")));
}

TEST(MeasurementCanonTest, CanonicalizedFastPath)
{
    Measurement m(3.0, xdataset::parse_unit("Hz"));  // already coherent SI
    Measurement c = m.canonicalized();
    EXPECT_DOUBLE_EQ(c.as_scalar<double>(), 3.0);
    EXPECT_TRUE(xdataset::same_dimension(c.unit(), xdataset::parse_unit("Hz")));
}

TEST(MeasurementCanonTest, CanonicalizedDegC)
{
    Measurement m(100.0, xdataset::parse_unit("degC"));
    Measurement c = m.canonicalized();
    EXPECT_NEAR(c.as_scalar<double>(), 373.15, 1e-10);
    EXPECT_FALSE(xdataset::is_affine(c.unit()));
}

TEST(MeasurementCanonTest, CanonicalizedStringNoValueChange)
{
    Measurement m(std::string("hello"), xdataset::parse_unit("m"));
    Measurement c = m.canonicalized();
    EXPECT_EQ(c.as_scalar<std::string>(), "hello");
    EXPECT_TRUE(xdataset::same_dimension(c.unit(), xdataset::parse_unit("m")));
}

// =========================================================================
