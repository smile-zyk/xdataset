#include "data_series.h"
#include "data_frame.h"

#include <gtest/gtest.h>

#include <complex>
#include <string>
#include <vector>


using xdataset::DataKind;
using xdataset::DataSeries;
using xdataset::Measurement;
using xdataset::DataType;
using xdataset::FormatOptions;
using xdataset::NumberFormat;
using xdataset::ComplexFormat;
using xdataset::Index;
using xdataset::MultiIndexSelector;
using xdataset::Unit;
using xdataset::VecXd;
using xdataset::VecXi;
using xdataset::VecXcd;
using xdataset::VecXs;
using xdataset::MatXd;
using xdataset::MatXi;
using xdataset::MatXcd;
using xdataset::VecXcd;
using xdataset::VecXs;
using xdataset::MatXd;
using xdataset::MatXi;
using xdataset::MatXcd;
using xdataset::MatXs;

// ---------------------------------------------------------------------------

// =========================================================================
//  Measurement -- construction, type queries, unit
// =========================================================================

TEST(CellTest, ScalarCellCreateAndMutate) {
    Measurement m = Measurement::Real(42.0);
    EXPECT_EQ(m.data_kind(), DataKind::kScalar);
    EXPECT_EQ(m.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(m.as_scalar<double>(), 42.0);

    m = Measurement::Real(3.5);
    EXPECT_DOUBLE_EQ(m.as_scalar<double>(), 3.5);
}

TEST(CellTest, IntegerCellDtype) {
    Measurement m = Measurement::Integer(7);
    EXPECT_EQ(m.data_type(), DataType::kInteger);
    EXPECT_EQ(m.as_scalar<int>(), 7);
}

TEST(CellTest, ComplexCellDtype) {
    using cd = std::complex<double>;
    Measurement m = Measurement::Complex(cd(1.0, 2.0));
    EXPECT_EQ(m.data_type(), DataType::kComplex);
    EXPECT_DOUBLE_EQ(m.as_scalar<cd>().real(), 1.0);
    EXPECT_DOUBLE_EQ(m.as_scalar<cd>().imag(), 2.0);
}

TEST(CellTest, StringCellDtype) {
    Measurement m = Measurement::String(std::string("hello"));
    EXPECT_EQ(m.data_type(), DataType::kString);
    EXPECT_EQ(m.as_scalar<std::string>(), "hello");
}

TEST(CellTest, AppendCellToSeries) {
    Measurement m = Measurement::Real(3.5);
    DataSeries s = DataSeries::CreateScalar<double>(0);
    s.append(Measurement::Real(1.25));
    s.append(m);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(1), 3.5);
}

TEST(CellTest, AppendTypePromotionIntToReal) {
    Measurement int_cell = Measurement::Integer(10);
    DataSeries s = DataSeries::CreateScalar<double>(0);
    EXPECT_THROW(s.append(int_cell), std::bad_cast);
}

TEST(CellTest, AppendTypePromotionIntToComplex) {
    Measurement int_cell = Measurement::Integer(10);
    DataSeries s = DataSeries::CreateScalar<std::complex<double>>(0);
    EXPECT_THROW(s.append(int_cell), std::bad_cast);
}

TEST(CellTest, AppendTypePromotionRealToComplex) {
    Measurement real_cell = Measurement::Real(3.5);
    DataSeries s = DataSeries::CreateScalar<std::complex<double>>(0);
    EXPECT_THROW(s.append(real_cell), std::bad_cast);
}

TEST(CellTest, AppendStillThrowsOnCompleteMismatch) {
    // Vector DataSeries, trying to append a different-shaped vector - still throws.
    VecXd v(4); v << 1., 2., 3., 4.;
    DataSeries s = DataSeries::CreateVector<double>(3, 0);
    EXPECT_THROW(s.append(Measurement::Vector(v)), std::bad_cast);
}

TEST(CellTest, AppendVectorShapeMismatchThrows) {
    VecXd vd(3); vd << 1., 2., 3.;
    DataSeries s(DataType::kComplex, xdataset::DataShape::Vector(2));
    EXPECT_THROW(s.append(Measurement::Vector(vd)), std::bad_cast);
}

TEST(CellTest, AppendUnitMismatchThrows) {
    // First append succeeds and sets the series unit.
    Measurement m_m = Measurement::Real(1.0).set_unit(xdataset::Unit::parse("meter"));
    DataSeries s = DataSeries::CreateScalar<double>(0);
    s.append(m_m);
    EXPECT_TRUE(s.unit().same_dimension(xdataset::Unit::parse("meter")));

    // Subsequent append with incompatible unit must throw.
    Measurement m_s = Measurement::Real(2.0).set_unit(xdataset::Unit::parse("sec"));
    EXPECT_THROW(s.append(m_s), std::invalid_argument);
}

TEST(CellTest, CellAtRoundtripScalar) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{1, 2, 3});
    Measurement m = s.measurement_at(1);
    EXPECT_EQ(m.data_kind(), DataKind::kScalar);
    EXPECT_EQ(m.data_type(), DataType::kInteger);
    EXPECT_EQ(m.as_scalar<int>(), 2);
}

TEST(CellTest, CellAtRoundtripVector) {
    DataSeries vecs(DataType::kReal, xdataset::DataShape::Vector(3));
    vecs.resize(2);
    vecs.vector_at<double>(0) << 1.0, 2.0, 3.0;
    vecs.vector_at<double>(1) << 4.0, 5.0, 6.0;
    Measurement m = vecs.measurement_at(1);
    EXPECT_EQ(m.data_kind(), DataKind::kVector);
    EXPECT_DOUBLE_EQ(m.as_vector<double>()(0), 4.0);
    EXPECT_DOUBLE_EQ(m.as_vector<double>()(2), 6.0);
}

TEST(CellTest, CellAtRoundtripMatrix) {
    DataSeries mats(DataType::kReal, xdataset::DataShape::Matrix(2, 2));
    mats.resize(1);
    mats.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    Measurement m = mats.measurement_at(0);
    EXPECT_EQ(m.data_kind(), DataKind::kMatrix);
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
    Measurement m = Measurement::Real(3.14);
    m.set_unit(xdataset::Unit::parse("meter"));
    Measurement m2(m);
    EXPECT_TRUE(m2.unit().same_dimension(m.unit()));
}

TEST(CellUnitTest, MovePropagatesUnit)
{
    Measurement m = Measurement::Real(2.72);
    m.set_unit(xdataset::Unit::parse("Hz"));
    Measurement m2(std::move(m));
    EXPECT_TRUE(m2.unit().same_dimension(xdataset::Unit::parse("Hz")));
}

TEST(CellUnitTest, AssignPropagatesUnit)
{
    Measurement m1 = Measurement::Real(1.0);
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
    Measurement m = Measurement::Real(5.0).set_unit(xdataset::Unit::parse("cm"));
    Measurement c = m.canonicalized();
    // 5 cm -> 0.05 m
    EXPECT_DOUBLE_EQ(c.as_scalar<double>(), 0.05);
    EXPECT_DOUBLE_EQ(c.unit().multiplier(), 1.0);
    EXPECT_TRUE(c.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(MeasurementCanonTest, CanonicalizedFastPath)
{
    Measurement m = Measurement::Real(3.0).set_unit(xdataset::Unit::parse("Hz"));  // already coherent SI
    Measurement c = m.canonicalized();
    EXPECT_DOUBLE_EQ(c.as_scalar<double>(), 3.0);
    EXPECT_TRUE(c.unit().same_dimension(xdataset::Unit::parse("Hz")));
}

TEST(MeasurementCanonTest, CanonicalizedStringNoValueChange)
{
    Measurement m = Measurement::String(std::string("hello")).set_unit(xdataset::Unit::parse("meter"));
    Measurement c = m.canonicalized();
    EXPECT_EQ(c.as_scalar<std::string>(), "hello");
    EXPECT_TRUE(c.unit().same_dimension(xdataset::Unit::parse("meter")));
}

// =========================================================================
//  Auto-scaling (best unit display) -- only kEngineering does this now
//
//  kFull (the default) never auto-scales, so these tests opt in explicitly.
// =========================================================================

namespace
{
FormatOptions engineering()
{
    FormatOptions o;
    o.number_format = NumberFormat::kEngineering;
    return o;
}
} // namespace

TEST(MeasurementFormatTest, AutoScaleMega)
{
    Measurement m = Measurement::Real(1e9).set_unit(xdataset::Unit::parse("Hz"));
    EXPECT_EQ(m.to_string(engineering()), "1 GHz");
}

TEST(MeasurementFormatTest, AutoScaleMilli)
{
    Measurement m = Measurement::Real(0.002).set_unit(xdataset::Unit::parse("V"));
    EXPECT_EQ(m.to_string(engineering()), "2 mV");
}

TEST(MeasurementFormatTest, AutoScaleKiloMeter)
{
    Measurement m = Measurement::Real(5000).set_unit(xdataset::Unit::parse("meter"));
    EXPECT_EQ(m.to_string(engineering()), "5 Kmeter");
}

TEST(MeasurementFormatTest, AutoScaleMilliMeter)
{
    Measurement m = Measurement::Real(0.003).set_unit(xdataset::Unit::parse("meter"));
    EXPECT_EQ(m.to_string(engineering()), "3 mmeter");
}

TEST(MeasurementFormatTest, AutoScaleNoneForDimensionless)
{
    Measurement m = Measurement::Real(3.14);
    // 3.14 is already in [1, 1000) -> no prefix.
    EXPECT_EQ(m.to_string(engineering()), "3.14");
}

TEST(MeasurementFormatTest, AutoScaleMegaDimensionless)
{
    Measurement m = Measurement::Integer(1000000);
    // Dimensionless values get the best-display prefix too: 1000000 -> "1 M".
    EXPECT_EQ(m.to_string(engineering()), "1 M");
}

TEST(MeasurementFormatTest, AutoScaleKiloFor100Hz)
{
    Measurement m = Measurement::Real(100.0).set_unit(xdataset::Unit::parse("Hz"));
    // 100 Hz stays 100 Hz (1 <= 100 < 1000).
    EXPECT_EQ(m.to_string(engineering()), "100 Hz");
}

// =========================================================================
//  Measurement::to_string(FormatOptions) -- NumberFormat
// =========================================================================

namespace
{
using xdataset::FormatDefaults;
using xdataset::FormatScope;
using xdataset::DisplayScale;
using xdataset::ResolveScale;
using xdataset::FormatWithScale;

const Unit kV  = Unit::parse("V");
const Unit kHz = Unit::parse("Hz");
const Unit kOhm = Unit::parse("Ohm");

std::string fmt(double v, const Unit& u, NumberFormat nf, int digits = 6)
{
    FormatOptions o;
    o.number_format = nf;
    o.significant_digits = digits;
    return Measurement::Real(v, u).to_string(o);
}
} // namespace

// kFull is the DEFAULT mode: no exponent, no unit auto-scaling.
TEST(MeasurementFormatOptionsTest, DefaultIsFull)
{
    FormatOptions o;
    EXPECT_EQ(o.number_format, NumberFormat::kFull);

    EXPECT_EQ(Measurement::Real(0.002, kV).to_string(),   "0.002 V");
    EXPECT_EQ(Measurement::Real(1500.0, kV).to_string(),  "1500 V");
    EXPECT_EQ(Measurement::Real(1e9, kHz).to_string(),    "1000000000 Hz");
    // An integer never gets a decimal point, even in Full mode.
    EXPECT_EQ(Measurement::Integer(42).to_string(),       "42");
    // Dimensionless: no prefix, because Full never auto-scales.
    EXPECT_EQ(Measurement::Real(0.002).to_string(),       "0.002");
    EXPECT_EQ(Measurement::Real(1e9).to_string(),         "1000000000");
}

TEST(MeasurementFormatOptionsTest, FullShowsAllIntegerDigitsNoScaling)
{
    // "Full": every digit before the decimal point, no exponent, and NO unit
    // auto-scaling (the exponent is carried by the digits).
    //
    // The significant-digit budget is spent on the integer digits first; any
    // remainder becomes decimals, and trailing zeros are NOT padded.
    // 1530000.123 has 7 integer digits, which already exceeds the 6-digit
    // budget, so no decimals are shown.
    EXPECT_EQ(fmt(1530000.123, kHz, NumberFormat::kFull), "1530000 Hz");
    EXPECT_EQ(fmt(1000.0, kHz, NumberFormat::kFull),      "1000 Hz");
    EXPECT_EQ(fmt(0.002, kV, NumberFormat::kFull),        "0.002 V");
}

TEST(MeasurementFormatOptionsTest, FullDoesNotPadTrailingZeros)
{
    // The digit budget caps precision but must NOT pad with zeros.
    EXPECT_EQ(fmt(3.14, kHz, NumberFormat::kFull),   "3.14 Hz");
    EXPECT_EQ(fmt(1.0, kHz, NumberFormat::kFull),    "1 Hz");
    EXPECT_EQ(fmt(42.0, kHz, NumberFormat::kFull),   "42 Hz");
    EXPECT_EQ(fmt(0.5, kV, NumberFormat::kFull),     "0.5 V");
    EXPECT_EQ(fmt(-2.0, kV, NumberFormat::kFull),    "-2 V");
    EXPECT_EQ(fmt(0.0, kV, NumberFormat::kFull),     "0 V");
    EXPECT_EQ(fmt(1.0 / 3.0, kHz, NumberFormat::kFull), "0.333333 Hz");
}

TEST(MeasurementFormatOptionsTest, FullHonoursSignificantDigits)
{
    EXPECT_EQ(fmt(1234.567890, kHz, NumberFormat::kFull, 3), "1235 Hz");
    EXPECT_EQ(fmt(1234.567890, kHz, NumberFormat::kFull, 6), "1234.57 Hz");
    EXPECT_EQ(fmt(1234.567890, kHz, NumberFormat::kFull, 9), "1234.56789 Hz");
}

TEST(MeasurementFormatOptionsTest, ScientificUsesExponentNoScaling)
{
    EXPECT_EQ(fmt(1000.0, kHz, NumberFormat::kScientific, 3), "1e3 Hz");
    EXPECT_EQ(fmt(1530000.0, kHz, NumberFormat::kScientific), "1.53e6 Hz");
    EXPECT_EQ(fmt(0.002, kV, NumberFormat::kScientific),      "2e-3 V");
}

TEST(MeasurementFormatOptionsTest, EngineeringForcesMultipleOfThree)
{
    EXPECT_EQ(fmt(1000.0, kHz, NumberFormat::kEngineering),  "1 KHz");
    EXPECT_EQ(fmt(1e9, kHz, NumberFormat::kEngineering),     "1 GHz");
    EXPECT_EQ(fmt(0.002, kV, NumberFormat::kEngineering),    "2 mV");
    EXPECT_EQ(fmt(4700.0, kOhm, NumberFormat::kEngineering), "4.7 KOhm");
}

TEST(MeasurementFormatOptionsTest, IntegerBases)
{
    EXPECT_EQ(fmt(255.0, kV, NumberFormat::kHex),    "0xff");
    EXPECT_EQ(fmt(8.0, kV, NumberFormat::kOctal),    "010");
    EXPECT_EQ(fmt(5.0, kV, NumberFormat::kBinary),   "0b101");
    // Non-integral values cannot be represented positionally -> decimal
    // (Full notation).
    EXPECT_EQ(fmt(1.5, kV, NumberFormat::kHex),      "1.5");
    // No unit suffix in base modes.
    EXPECT_EQ(fmt(255.0, kV, NumberFormat::kHex),    "0xff");
}

// =========================================================================
//  show_unit
// =========================================================================

TEST(MeasurementFormatOptionsTest, ShowUnitFalseIsBareInFull)
{
    FormatOptions o;   // kFull (default)
    o.show_unit = false;
    // Full never auto-scales anyway, so hiding the unit only drops the suffix.
    EXPECT_EQ(Measurement::Real(0.002, kV).to_string(o),  "0.002");
    EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(o), "2400000000");
}

TEST(MeasurementFormatOptionsTest, ShowUnitFalseKeepsSpicePrefixInEngineering)
{
    // The 10^3 step belongs to the MODE, not to the unit, so the prefix
    // survives -- SPICE style, no space.
    FormatOptions o;
    o.number_format = NumberFormat::kEngineering;
    o.show_unit = false;
    EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(o),  "2.4G");
    EXPECT_EQ(Measurement::Real(1000.0, kHz).to_string(o), "1K");
    EXPECT_EQ(Measurement::Real(0.002, kV).to_string(o),   "2m");
    EXPECT_EQ(Measurement::Real(4700.0, kOhm).to_string(o), "4.7K");
    // No scaling needed -> no prefix at all.
    EXPECT_EQ(Measurement::Real(50.0, kHz).to_string(o),   "50");
}

TEST(MeasurementFormatOptionsTest, ShowUnitTrueIsDefaultBehaviour)
{
    FormatOptions o;
    EXPECT_TRUE(o.show_unit);
    o.number_format = NumberFormat::kEngineering;
    EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(o), "2.4 GHz");
}

// =========================================================================
//  ComplexFormat
// =========================================================================

namespace
{
std::string cfmt(const std::complex<double>& z, const Unit& u, ComplexFormat cf)
{
    FormatOptions o;
    o.complex_format = cf;
    return Measurement::Complex(z, u).to_string(o);
}
} // namespace

TEST(MeasurementFormatOptionsTest, ComplexRealImaginary)
{
    // kRealImaginary is the ONLY mode that carries a unit.  Under the default
    // kFull there is no auto-scaling and no zero padding, so the value stays
    // in volts with its natural digits.
    const std::string s = cfmt(std::complex<double>(0.778, -0.258), kV,
                               ComplexFormat::kRealImaginary);
    EXPECT_EQ(s, "0.778-0.258i V");
}

// With kEngineering, kRealImaginary scales both parts by ONE shared factor
// (resolved from |z|) and shows the unit.
TEST(MeasurementFormatOptionsTest, ComplexRealImaginaryEngineering)
{
    FormatOptions o;
    o.number_format = NumberFormat::kEngineering;
    o.complex_format = ComplexFormat::kRealImaginary;
    const std::string s =
        Measurement::Complex(std::complex<double>(0.778, -0.258), kV).to_string(o);
    EXPECT_EQ(s, "778-258i mV");
}

TEST(MeasurementFormatOptionsTest, ComplexMagDegrees)
{
    // Pure "a/b" pair: NO unit, NO scaling.
    const std::string s = cfmt(std::complex<double>(0.778, -0.258), kV,
                               ComplexFormat::kMagDegrees);
    EXPECT_EQ(s, "0.819663/-18.3465");
    EXPECT_TRUE(s.find("V") == std::string::npos);
    EXPECT_TRUE(s.find("mV") == std::string::npos);
}

TEST(MeasurementFormatOptionsTest, ComplexDbDegrees)
{
    // 20*log10(0.8197) = -1.727 dB ; arg = -18.35 deg.  No unit.
    const std::string s = cfmt(std::complex<double>(0.778, -0.258), kV,
                               ComplexFormat::kDbDegrees);
    EXPECT_EQ(s, "-1.72729/-18.3465");
    EXPECT_TRUE(s.find("V") == std::string::npos);
}

TEST(MeasurementFormatOptionsTest, ComplexRadiansVariants)
{
    const std::string mag = cfmt(std::complex<double>(0.778, -0.258), kV,
                                 ComplexFormat::kMagRadians);
    EXPECT_EQ(mag, "0.819663/-0.320207");

    const std::string db = cfmt(std::complex<double>(0.778, -0.258), kV,
                                ComplexFormat::kDbRadians);
    EXPECT_EQ(db, "-1.72729/-0.320207");
}

// Only kRealImaginary carries a unit; every other mode must be unit-less.
TEST(MeasurementFormatOptionsTest, OnlyRealImaginaryCarriesUnit)
{
    const std::complex<double> z(0.778, -0.258);
    const std::string ri = cfmt(z, kV, ComplexFormat::kRealImaginary);
    EXPECT_TRUE(ri.find("V") != std::string::npos);

    const ComplexFormat others[] = {
        ComplexFormat::kMagDegrees, ComplexFormat::kDbDegrees,
        ComplexFormat::kMagRadians, ComplexFormat::kDbRadians};
    for (ComplexFormat cf : others)
    {
        const std::string s = cfmt(z, kV, cf);
        EXPECT_TRUE(s.find("V") == std::string::npos) << s;
        EXPECT_TRUE(s.find("mV") == std::string::npos) << s;
        EXPECT_TRUE(s.find('/') != std::string::npos) << s;
    }
}

TEST(MeasurementFormatOptionsTest, ComplexDefaultIsRealImaginary)
{
    FormatOptions o;
    EXPECT_EQ(o.complex_format, ComplexFormat::kRealImaginary);
}

// =========================================================================
//  significant_digits clamping + shared-instance safety
// =========================================================================

TEST(MeasurementFormatOptionsTest, SignificantDigitsClamped)
{
    // Clamped to [1, 17]; must not crash or produce garbage.
    FormatOptions o;
    o.number_format = NumberFormat::kFull;
    o.significant_digits = 0;
    EXPECT_FALSE(Measurement::Real(1234.5, kHz).to_string(o).empty());
    o.significant_digits = 100;
    EXPECT_FALSE(Measurement::Real(1234.5, kHz).to_string(o).empty());
}

// Rendering is a pure function: interleaving different option sets across
// calls must not leak state between them.
TEST(MeasurementFormatOptionsTest, OptionsDoNotLeakBetweenCalls)
{
    FormatOptions a;                                   // kFull (default)
    FormatOptions b; b.number_format = NumberFormat::kEngineering;
    b.show_unit = false;

    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(a), "2400000000 Hz");
        EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(b), "2.4G");
        EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(),  "2400000000 Hz");
    }
}

// =========================================================================
//  FormatDefaults / FormatScope
// =========================================================================

TEST(FormatDefaultsTest, DefaultsToFull)
{
    // Reset first so this test does not depend on execution order.
    FormatDefaults::Instance().Set(FormatOptions());
    EXPECT_EQ(Measurement::Real(0.002, kV).to_string(), "0.002 V");
}

TEST(FormatDefaultsTest, SetAffectsTo_stringWithoutOptions)
{
    FormatOptions o;
    o.number_format = NumberFormat::kEngineering;

    FormatDefaults::Instance().Set(o);
    EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(), "2.4 GHz");

    // Explicit options still override the global default.
    FormatOptions full;
    EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(full),
              "2400000000 Hz");

    FormatDefaults::Instance().Set(FormatOptions());   // restore
}

// to_string(opts) must NOT disturb the process-wide default.
TEST(FormatDefaultsTest, ExplicitOptionsDoNotChangeTheDefault)
{
    FormatDefaults::Instance().Set(FormatOptions());   // kFull

    FormatOptions eng;
    eng.number_format = NumberFormat::kEngineering;
    EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(eng), "2.4 GHz");

    // The default is untouched.
    EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(), "2400000000 Hz");
}

TEST(FormatDefaultsTest, ScopeRestoresOnExit)
{
    FormatDefaults::Instance().Set(FormatOptions());   // known start state

    FormatOptions eng;
    eng.number_format = NumberFormat::kEngineering;

    {
        FormatScope scope(eng);
        EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(), "2.4 GHz");
    }
    // Restored to what was in effect before the scope (kFull).
    EXPECT_EQ(Measurement::Real(2.4e9, kHz).to_string(), "2400000000 Hz");
}

TEST(FormatDefaultsTest, ScopeIsNestable)
{
    FormatDefaults::Instance().Set(FormatOptions());   // known start state

    FormatOptions eng;
    eng.number_format = NumberFormat::kEngineering;
    FormatOptions sci;
    sci.number_format = NumberFormat::kScientific;

    {
        FormatScope outer(eng);
        EXPECT_EQ(Measurement::Real(1000.0, kHz).to_string(), "1 KHz");
        {
            FormatScope inner(sci);
            EXPECT_EQ(Measurement::Real(1000.0, kHz).to_string(), "1e3 Hz");
        }
        // Inner scope restored the outer one, not the process default.
        EXPECT_EQ(Measurement::Real(1000.0, kHz).to_string(), "1 KHz");
    }
    // Back to the process default (kFull, no auto-scaling).
    EXPECT_EQ(Measurement::Real(1000.0, kHz).to_string(), "1000 Hz");
}

// =========================================================================
//  ResolveScale / FormatWithScale -- the host-side hoisting seam
// =========================================================================

TEST(FormatScaleTest, ResolveThenFormatMatchesDirectRender)
{
    FormatOptions o;
    o.number_format = NumberFormat::kEngineering;

    // Hoist the expensive unit resolution out of the loop.
    const DisplayScale s = ResolveScale(2.4e9, kHz, o);

    EXPECT_EQ(FormatWithScale(2.4e9, s, o.number_format, o.significant_digits),
              Measurement::Real(2.4e9, kHz).to_string(o));
}

TEST(FormatScaleTest, SuffixCarriesLeadingSpaceForUnits)
{
    FormatOptions o;
    o.number_format = NumberFormat::kEngineering;
    const DisplayScale s = ResolveScale(2.4e9, kHz, o);
    EXPECT_EQ(s.suffix, " GHz");
}

TEST(FormatScaleTest, SuffixIsSpiceBareWhenUnitHidden)
{
    FormatOptions o;
    o.number_format = NumberFormat::kEngineering;
    o.show_unit = false;
    const DisplayScale s = ResolveScale(2.4e9, kHz, o);
    EXPECT_EQ(s.suffix, "G");       // no leading space: SPICE style
}

TEST(FormatScaleTest, FullModeDoesNotScale)
{
    FormatOptions o;                 // kFull
    const DisplayScale s = ResolveScale(2.4e9, kHz, o);
    EXPECT_DOUBLE_EQ(s.scale, 1.0);
    EXPECT_EQ(s.suffix, " Hz");
}

TEST(FormatScaleTest, IntegerWithScaleStaysIntegralWhenPossible)
{
    FormatOptions o;                 // kFull, no scaling
    const DisplayScale s = ResolveScale(42.0, Unit(), o);
    // An integer must never gain a decimal point.
    EXPECT_EQ(FormatWithScale(42, s, o.number_format, o.significant_digits),
              "42");
}

// =========================================================================
//  Measurement: to_dataframe
// =========================================================================

TEST(MeasurementToDataFrameTest, Scalar)
{
    Measurement m = Measurement::Real(3.14).set_unit(xdataset::Unit::parse("meter"));
    auto df = m.to_dataframe("distance");

    EXPECT_EQ(df->row_count(), 1u);
    ASSERT_EQ(df->headers().size(), 1u);
    EXPECT_EQ(df->headers()[0], "distance");
    EXPECT_EQ(df->GetRow(0).fields[0].to_string(), "3.14 meter");
}

TEST(MeasurementToDataFrameTest, Vector)
{
    VecXd v(3);
    v << 1.0, 2.0, 3.0;
    Measurement m = Measurement::Vector(v);

    auto df = m.to_dataframe("pos");

    EXPECT_EQ(df->row_count(), 1u);
    ASSERT_EQ(df->headers().size(), 3u);
    EXPECT_EQ(df->headers()[0], "pos(1)");
    EXPECT_EQ(df->headers()[1], "pos(2)");
    EXPECT_EQ(df->headers()[2], "pos(3)");

    EXPECT_EQ(df->GetRow(0).fields[0].to_string(), "1");
    EXPECT_EQ(df->GetRow(0).fields[1].to_string(), "2");
    EXPECT_EQ(df->GetRow(0).fields[2].to_string(), "3");
}

TEST(MeasurementToDataFrameTest, Matrix)
{
    xdataset::MatXd mat(2, 2);
    mat << 1.0, 2.0, 3.0, 4.0;
    Measurement m = Measurement::Matrix(mat);

    auto df = m.to_dataframe("mat");

    EXPECT_EQ(df->row_count(), 1u);
    ASSERT_EQ(df->headers().size(), 4u);
    EXPECT_EQ(df->headers()[0], "mat(1,1)");
    EXPECT_EQ(df->headers()[1], "mat(1,2)");
    EXPECT_EQ(df->headers()[2], "mat(2,1)");
    EXPECT_EQ(df->headers()[3], "mat(2,2)");

    EXPECT_EQ(df->GetRow(0).fields[0].to_string(), "1");
    EXPECT_EQ(df->GetRow(0).fields[1].to_string(), "2");
    EXPECT_EQ(df->GetRow(0).fields[2].to_string(), "3");
    EXPECT_EQ(df->GetRow(0).fields[3].to_string(), "4");
}

TEST(MeasurementToDataFrameTest, ToCsvRoundtrip)
{
    VecXd v(2);
    v << 10.0, 20.0;
    Measurement m = Measurement::Vector(v);

    auto df = m.to_dataframe("velocity");
    const std::string csv = df->ToCsv();

    // Header row
    EXPECT_NE(csv.find(",velocity(1),velocity(2)"), std::string::npos);
    // Data row -> single measurement, no multi-index
    //   FormatMultiIndex() gives "[]", EscapeCsvField does not quote it
    //   (no comma / quote / newline characters), so: [],10,20
    EXPECT_NE(csv.find("0,10,20"), std::string::npos);
}

// =========================================================================
//  Measurement::at
// =========================================================================

TEST(MeasurementAtTest, ScalarThrows)
{
    Measurement m = Measurement::Real(42.0);
    EXPECT_THROW(m.at({MultiIndexSelector::Any()}), std::logic_error);
}

TEST(MeasurementAtTest, VectorAtEqualReturnsScalar)
{
    VecXd v(4); v << 10.0, 20.0, 30.0, 40.0;
    Measurement m = Measurement::Vector(v);

    Measurement result = m.at({MultiIndexSelector::Equal(2)});
    ASSERT_EQ(result.data_kind(), DataKind::kScalar);
    EXPECT_DOUBLE_EQ(result.as_scalar<double>(), 30.0);
}

TEST(MeasurementAtTest, VectorAtInReturnsSubVector)
{
    VecXd v(5); v << 1.0, 2.0, 3.0, 4.0, 5.0;
    Measurement m = Measurement::Vector(v);

    Measurement result = m.at({MultiIndexSelector::In({0, 2, 4})});
    ASSERT_EQ(result.data_kind(), DataKind::kVector);
    auto vec = result.as_vector<double>();
    EXPECT_EQ(vec.size(), 3);
    EXPECT_DOUBLE_EQ(vec(0), 1.0);
    EXPECT_DOUBLE_EQ(vec(1), 3.0);
    EXPECT_DOUBLE_EQ(vec(2), 5.0);
}

TEST(MeasurementAtTest, VectorAtInPreservesUnit)
{
    Unit u = Unit::parse("V");
    VecXd v(3); v << 1.0, 2.0, 3.0;
    Measurement m = Measurement::Vector(v).set_unit(u);

    Measurement result = m.at({MultiIndexSelector::In({1, 2})});
    EXPECT_TRUE(result.unit().same_dimension(u));
}

TEST(MeasurementAtTest, VectorAtAnyReturnsAll)
{
    VecXd v(3); v << 1.0, 2.0, 3.0;
    Measurement m = Measurement::Vector(v);

    Measurement result = m.at({MultiIndexSelector::Any()});
    ASSERT_EQ(result.data_kind(), DataKind::kVector);
    auto vec = result.as_vector<double>();
    EXPECT_EQ(vec.size(), 3);
}

TEST(MeasurementAtTest, MatrixAtSingleElementReturnsScalar)
{
    MatXd mat(2, 3);
    mat << 1.0, 2.0, 3.0,
           4.0, 5.0, 6.0;
    Measurement m = Measurement::Matrix(mat);

    Measurement result = m.at({MultiIndexSelector::Equal(1), MultiIndexSelector::Equal(2)});
    ASSERT_EQ(result.data_kind(), DataKind::kScalar);
    EXPECT_DOUBLE_EQ(result.as_scalar<double>(), 6.0);
}

TEST(MeasurementAtTest, MatrixAtSingleRowReturnsVector)
{
    MatXd mat(3, 3);
    mat << 1.0, 2.0, 3.0,
           4.0, 5.0, 6.0,
           7.0, 8.0, 9.0;
    Measurement m = Measurement::Matrix(mat);

    Measurement result = m.at({MultiIndexSelector::Equal(1), MultiIndexSelector::Any()});
    ASSERT_EQ(result.data_kind(), DataKind::kVector);
    auto vec = result.as_vector<double>();
    EXPECT_EQ(vec.size(), 3);
    EXPECT_DOUBLE_EQ(vec(0), 4.0);
    EXPECT_DOUBLE_EQ(vec(1), 5.0);
    EXPECT_DOUBLE_EQ(vec(2), 6.0);
}

TEST(MeasurementAtTest, MatrixAtSingleColumnReturnsVector)
{
    MatXd mat(3, 2);
    mat << 1.0, 2.0,
           3.0, 4.0,
           5.0, 6.0;
    Measurement m = Measurement::Matrix(mat);

    Measurement result = m.at({MultiIndexSelector::Any(), MultiIndexSelector::Equal(0)});
    ASSERT_EQ(result.data_kind(), DataKind::kVector);
    auto vec = result.as_vector<double>();
    EXPECT_EQ(vec.size(), 3);
    EXPECT_DOUBLE_EQ(vec(0), 1.0);
    EXPECT_DOUBLE_EQ(vec(1), 3.0);
    EXPECT_DOUBLE_EQ(vec(2), 5.0);
}

TEST(MeasurementAtTest, MatrixAtInReturnsSubMatrix)
{
    MatXd mat(4, 4);
    mat << 1.0,  2.0,  3.0,  4.0,
           5.0,  6.0,  7.0,  8.0,
           9.0,  10.0, 11.0, 12.0,
           13.0, 14.0, 15.0, 16.0;
    Measurement m = Measurement::Matrix(mat);

    Measurement result = m.at({MultiIndexSelector::In({0, 2}), MultiIndexSelector::In({1, 3})});
    ASSERT_EQ(result.data_kind(), DataKind::kMatrix);
    auto sub = result.as_matrix<double>();
    EXPECT_EQ(sub.rows(), 2);
    EXPECT_EQ(sub.cols(), 2);
    EXPECT_DOUBLE_EQ(sub(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(sub(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(sub(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(sub(1, 1), 12.0);
}

TEST(MeasurementAtTest, MatrixAtPreservesUnit)
{
    Unit u = Unit::parse("V");
    MatXd mat(2, 2);
    mat << 1.0, 2.0, 3.0, 4.0;
    Measurement m = Measurement::Matrix(mat).set_unit(u);

    Measurement result = m.at({MultiIndexSelector::Equal(0), MultiIndexSelector::Equal(0)});
    ASSERT_EQ(result.data_kind(), DataKind::kScalar);
    EXPECT_DOUBLE_EQ(result.as_scalar<double>(), 1.0);
    EXPECT_TRUE(result.unit().same_dimension(u));
}

TEST(MeasurementAtTest, IntegerVector)
{
    VecXi v(3); v << 10, 20, 30;
    Measurement m = Measurement::Vector(v);

    Measurement result = m.at({MultiIndexSelector::In({0, 2})});
    ASSERT_EQ(result.data_kind(), DataKind::kVector);
    auto vec = result.as_vector<int>();
    EXPECT_EQ(vec(0), 10);
    EXPECT_EQ(vec(1), 30);
}

TEST(MeasurementAtTest, ComplexMatrix)
{
    MatXcd mat(2, 2);
    mat << std::complex<double>(1, 0), std::complex<double>(2, 0),
           std::complex<double>(3, 0), std::complex<double>(4, 0);
    Measurement m = Measurement::Matrix(mat);

    Measurement result = m.at({MultiIndexSelector::Any(), MultiIndexSelector::Equal(1)});
    ASSERT_EQ(result.data_kind(), DataKind::kVector);
    auto vec = result.as_vector<std::complex<double>>();
    EXPECT_DOUBLE_EQ(vec(0).real(), 2.0);
    EXPECT_DOUBLE_EQ(vec(1).real(), 4.0);
}

TEST(MeasurementAtTest, StringVector)
{
    VecXs v(3);
    v(0) = "a"; v(1) = "b"; v(2) = "c";
    Measurement m = Measurement::Vector(v);

    Measurement result = m.at({MultiIndexSelector::In({1, 2})});
    ASSERT_EQ(result.data_kind(), DataKind::kVector);
    auto vec = result.as_vector<std::string>();
    EXPECT_EQ(vec(0), "b");
    EXPECT_EQ(vec(1), "c");
}

TEST(MeasurementAtTest, TooManySelectorsThrows)
{
    VecXd v(3); v << 1.0, 2.0, 3.0;
    Measurement m = Measurement::Vector(v);
    EXPECT_THROW(m.at({MultiIndexSelector::Any(), MultiIndexSelector::Any()}), std::invalid_argument);
}

// =========================================================================
//  Measurement::operator==
// =========================================================================

TEST(MeasurementEqualityTest, ScalarEquals)
{
    EXPECT_TRUE(Measurement::Real(1.0) == Measurement::Real(1.0));
    EXPECT_FALSE(Measurement::Real(1.0) == Measurement::Real(2.0));
    EXPECT_TRUE(Measurement::Integer(5) == Measurement::Integer(5));
    EXPECT_FALSE(Measurement::Integer(5) == Measurement::Integer(6));
    EXPECT_TRUE(Measurement::String("abc") == Measurement::String("abc"));
    EXPECT_FALSE(Measurement::String("abc") == Measurement::String("abd"));
    EXPECT_TRUE(Measurement::Boolean(true) == Measurement::Boolean(true));
    EXPECT_FALSE(Measurement::Boolean(true) == Measurement::Boolean(false));
    EXPECT_TRUE(Measurement::Complex(std::complex<double>(1.0, 2.0)) ==
                Measurement::Complex(std::complex<double>(1.0, 2.0)));
    EXPECT_FALSE(Measurement::Complex(std::complex<double>(1.0, 2.0)) ==
                 Measurement::Complex(std::complex<double>(1.0, 3.0)));
}

TEST(MeasurementEqualityTest, DifferentDtypeNotEqual)
{
    EXPECT_FALSE(Measurement::Real(1.0) == Measurement::Integer(1));
    EXPECT_FALSE(Measurement::Integer(1) == Measurement::Real(1.0));
}

TEST(MeasurementEqualityTest, VectorEqualsElementwise)
{
    VecXd a(3); a << 1.0, 2.0, 3.0;
    VecXd b(3); b << 1.0, 2.0, 3.0;
    VecXd c(3); c << 1.0, 2.0, 9.0;
    EXPECT_TRUE(Measurement::Vector(a) == Measurement::Vector(b));
    EXPECT_FALSE(Measurement::Vector(a) == Measurement::Vector(c));
}

TEST(MeasurementEqualityTest, VectorShapeMismatchNotEqual)
{
    VecXd a(2); a << 1.0, 2.0;
    VecXd b(3); b << 1.0, 2.0, 3.0;
    EXPECT_FALSE(Measurement::Vector(a) == Measurement::Vector(b));
}

TEST(MeasurementEqualityTest, StringVectorEquals)
{
    VecXs a(2); a(0) = "x"; a(1) = "y";
    VecXs b(2); b(0) = "x"; b(1) = "y";
    VecXs c(2); c(0) = "x"; c(1) = "z";
    EXPECT_TRUE(Measurement::Vector(a) == Measurement::Vector(b));
    EXPECT_FALSE(Measurement::Vector(a) == Measurement::Vector(c));
}

TEST(MeasurementEqualityTest, MatrixEqualsElementwise)
{
    MatXd a(2, 2); a << 1.0, 2.0, 3.0, 4.0;
    MatXd b(2, 2); b << 1.0, 2.0, 3.0, 4.0;
    MatXd c(2, 2); c << 1.0, 2.0, 3.0, 5.0;
    EXPECT_TRUE(Measurement::Matrix(a) == Measurement::Matrix(b));
    EXPECT_FALSE(Measurement::Matrix(a) == Measurement::Matrix(c));
}

TEST(MeasurementEqualityTest, UnitIsIgnoredInEquality)
{
    // Value equality ignores the unit (a value equals the same value with any
    // unit); use unit().same_dimension() to compare units.
    const Unit volt = Unit::parse("V");
    EXPECT_TRUE(Measurement::Real(1.0, volt) == Measurement::Real(1.0, volt));
    EXPECT_FALSE(Measurement::Real(1.0, volt) == Measurement::Real(2.0, volt));
    // Same value, different unit -> still equal (value-only comparison).
    EXPECT_TRUE(Measurement::Real(1.0, volt) == Measurement::Real(1.0));
    // Different value -> not equal regardless of unit.
    EXPECT_FALSE(Measurement::Real(1.0, volt) == Measurement::Real(2.0));
}

// =========================================================================
//  Measurement::transform
// =========================================================================

TEST(MeasurementTransformTest, ScalarSquare) {
    Measurement m = Measurement::Real(3.0);
    Measurement result = m.transform([](double x) { return x * x; });
    EXPECT_EQ(result.data_kind(), DataKind::kScalar);
    EXPECT_EQ(result.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(result.as_scalar<double>(), 9.0);
}

TEST(MeasurementTransformTest, ScalarIntNegate) {
    Measurement m = Measurement::Integer(5);
    Measurement result = m.transform([](int x) { return -x; });
    EXPECT_EQ(result.data_kind(), DataKind::kScalar);
    EXPECT_EQ(result.data_type(), DataType::kInteger);
    EXPECT_EQ(result.as_scalar<int>(), -5);
}

TEST(MeasurementTransformTest, ScalarComplexAbsToReal) {
    Measurement m = Measurement::Complex(std::complex<double>(3.0, 4.0));
    Measurement result = m.transform([](std::complex<double> x) { return std::abs(x); });
    EXPECT_EQ(result.data_kind(), DataKind::kScalar);
    EXPECT_EQ(result.data_type(), DataType::kReal);
    EXPECT_DOUBLE_EQ(result.as_scalar<double>(), 5.0);
}

TEST(MeasurementTransformTest, ScalarDoubleToInt) {
    Measurement m = Measurement::Real(2.7);
    Measurement result = m.transform([](double x) { return static_cast<int>(x); });
    EXPECT_EQ(result.data_kind(), DataKind::kScalar);
    EXPECT_EQ(result.data_type(), DataType::kInteger);
    EXPECT_EQ(result.as_scalar<int>(), 2);
}

TEST(MeasurementTransformTest, ScalarString) {
    Measurement m = Measurement::String("hello");
    Measurement result = m.transform([](const std::string& s) { return s + "!"; });
    EXPECT_EQ(result.data_kind(), DataKind::kScalar);
    EXPECT_EQ(result.data_type(), DataType::kString);
    EXPECT_EQ(result.as_scalar<std::string>(), "hello!");
}

TEST(MeasurementTransformTest, VectorSquare) {
    VecXd v(3); v << 1.0, 2.0, 3.0;
    Measurement m = Measurement::Vector(v);
    Measurement result = m.transform([](double x) { return x * x; });
    EXPECT_EQ(result.data_kind(), DataKind::kVector);
    EXPECT_EQ(result.data_type(), DataType::kReal);
    auto vec = result.as_vector<double>();
    EXPECT_DOUBLE_EQ(vec(0), 1.0);
    EXPECT_DOUBLE_EQ(vec(1), 4.0);
    EXPECT_DOUBLE_EQ(vec(2), 9.0);
}

TEST(MeasurementTransformTest, VectorIntNegate) {
    VecXi v(3); v << 1, -2, 3;
    Measurement m = Measurement::Vector(v);
    Measurement result = m.transform([](int x) { return -x; });
    EXPECT_EQ(result.data_kind(), DataKind::kVector);
    EXPECT_EQ(result.data_type(), DataType::kInteger);
    auto vec = result.as_vector<int>();
    EXPECT_EQ(vec(0), -1);
    EXPECT_EQ(vec(1), 2);
    EXPECT_EQ(vec(2), -3);
}

TEST(MeasurementTransformTest, MatrixSquare) {
    MatXd m(2, 2);
    m << 1.0, 2.0, 3.0, 4.0;
    Measurement meas = Measurement::Matrix(m);
    Measurement result = meas.transform([](double x) { return x * x; });
    EXPECT_EQ(result.data_kind(), DataKind::kMatrix);
    EXPECT_EQ(result.data_type(), DataType::kReal);
    auto mat = result.as_matrix<double>();
    EXPECT_DOUBLE_EQ(mat(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mat(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(mat(1, 0), 9.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 16.0);
}

TEST(MeasurementTransformTest, MatrixComplexAbsToReal) {
    MatXcd m(2, 2);
    m << std::complex<double>(3.0, 4.0), std::complex<double>(0.0, -1.0),
         std::complex<double>(-5.0, 0.0), std::complex<double>(1.0, 1.0);
    Measurement meas = Measurement::Matrix(m);
    Measurement result = meas.transform([](std::complex<double> x) { return std::abs(x); });
    EXPECT_EQ(result.data_kind(), DataKind::kMatrix);
    EXPECT_EQ(result.data_type(), DataType::kReal);
    auto mat = result.as_matrix<double>();
    EXPECT_DOUBLE_EQ(mat(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(mat(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(mat(1, 0), 5.0);
    EXPECT_NEAR(mat(1, 1), std::sqrt(2.0), 1e-12);
}

// =========================================================================
