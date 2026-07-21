#include "unit.h"

#include <gtest/gtest.h>

using xdataset::Unit;

namespace xdataset
{

// =========================================================================
//  Construction & parse
// =========================================================================

TEST(UnitTest, DefaultIsDimensionlessUnit)
{
    Unit u;
    EXPECT_FALSE(u.has_dimension());
    EXPECT_TRUE(u.is_dimensionless());
    EXPECT_DOUBLE_EQ(u.multiplier(), 1.0);
}

TEST(UnitTest, ParseHz)
{
    Unit u = Unit::parse("Hz");
    EXPECT_TRUE(u.has_dimension());
    EXPECT_DOUBLE_EQ(u.multiplier(), 1.0);
}

TEST(UnitTest, ParseMeter)
{
    Unit u = Unit::parse("meter");
    EXPECT_TRUE(u.has_dimension());
    EXPECT_DOUBLE_EQ(u.multiplier(), 1.0);
}

TEST(UnitTest, ParseInvalidThrows)
{
    EXPECT_THROW(Unit::parse("blarghzzz"), std::invalid_argument);
    EXPECT_THROW(Unit::parse("xyz"), std::invalid_argument);
    EXPECT_THROW(Unit::parse("Pa"), std::invalid_argument);
    EXPECT_THROW(Unit::parse("km"), std::invalid_argument);
}

TEST(UnitTest, ParseEmptyThrows)
{
    EXPECT_THROW(Unit::parse(""), std::invalid_argument);
}

// =========================================================================
//  Scale factors
// =========================================================================

TEST(UnitTest, ScaleFactorT)  { EXPECT_DOUBLE_EQ(Unit::parse("T").multiplier(),  1e12); }
TEST(UnitTest, ScaleFactorG)  { EXPECT_DOUBLE_EQ(Unit::parse("G").multiplier(),  1e9);  }
TEST(UnitTest, ScaleFactorM)  { EXPECT_DOUBLE_EQ(Unit::parse("M").multiplier(),  1e6);  }
TEST(UnitTest, ScaleFactorK)  { EXPECT_DOUBLE_EQ(Unit::parse("K").multiplier(),  1e3);  }
TEST(UnitTest, ScaleFactork)  { EXPECT_DOUBLE_EQ(Unit::parse("k").multiplier(),  1e3);  }
TEST(UnitTest, ScaleFactor_u) { EXPECT_DOUBLE_EQ(Unit::parse("_").multiplier(),  1.0);  }
TEST(UnitTest, ScaleFactorm)  { EXPECT_DOUBLE_EQ(Unit::parse("m").multiplier(),  1e-3); }
TEST(UnitTest, ScaleFactoru)  { EXPECT_DOUBLE_EQ(Unit::parse("u").multiplier(),  1e-6); }
TEST(UnitTest, ScaleFactorn)  { EXPECT_DOUBLE_EQ(Unit::parse("n").multiplier(),  1e-9); }
TEST(UnitTest, ScaleFactorp)  { EXPECT_DOUBLE_EQ(Unit::parse("p").multiplier(),  1e-12);}
TEST(UnitTest, ScaleFactorf)  { EXPECT_DOUBLE_EQ(Unit::parse("f").multiplier(),  1e-15);}
TEST(UnitTest, ScaleFactora)  { EXPECT_DOUBLE_EQ(Unit::parse("a").multiplier(),  1e-18);}

TEST(UnitTest, ScaleFactorMIsDimensionless)
{
    Unit u = Unit::parse("M");
    EXPECT_FALSE(u.has_dimension());
    EXPECT_FALSE(u.is_dimensionless());
}

TEST(UnitTest, ScaleFactor_uIsDimlessAndUnit)
{
    Unit u = Unit::parse("_");
    EXPECT_FALSE(u.has_dimension());
    EXPECT_TRUE(u.is_dimensionless());
}

// =========================================================================
//  Scale factor + unit
// =========================================================================

TEST(UnitTest, MHz)
{
    Unit u = Unit::parse("MHz");
    EXPECT_TRUE(u.same_dimension(Unit::parse("Hz")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1e6);
}

TEST(UnitTest, kHz)
{
    Unit u = Unit::parse("kHz");
    EXPECT_TRUE(u.same_dimension(Unit::parse("Hz")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1e3);
}

TEST(UnitTest, kOhm)
{
    Unit u = Unit::parse("kOhm");
    EXPECT_TRUE(u.same_dimension(Unit::parse("Ohm")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1e3);
}

TEST(UnitTest, kV)
{
    Unit u = Unit::parse("kV");
    EXPECT_TRUE(u.same_dimension(Unit::parse("V")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1e3);
}

TEST(UnitTest, mA)
{
    Unit u = Unit::parse("mA");
    EXPECT_TRUE(u.same_dimension(Unit::parse("A")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1e-3);
}

TEST(UnitTest, uF)
{
    Unit u = Unit::parse("uF");
    EXPECT_TRUE(u.same_dimension(Unit::parse("F")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1e-6);
}

// =========================================================================
//  Predefined scaled units
// =========================================================================

TEST(UnitTest, Mil)
{
    Unit u = Unit::parse("mil");
    EXPECT_TRUE(u.same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 2.54e-5);
}

TEST(UnitTest, Mils)
{
    Unit u = Unit::parse("mils");
    EXPECT_TRUE(u.same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 2.54e-5);
}

TEST(UnitTest, In)
{
    Unit u = Unit::parse("in");
    EXPECT_TRUE(u.same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 2.54e-2);
}

TEST(UnitTest, Ft)
{
    Unit u = Unit::parse("ft");
    EXPECT_TRUE(u.same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 12 * 2.54e-2);
}

TEST(UnitTest, Mi)
{
    Unit u = Unit::parse("mi");
    EXPECT_TRUE(u.same_dimension(Unit::parse("meter")));
    EXPECT_NEAR(u.multiplier(), 1609.344, 0.01);
}

TEST(UnitTest, Cm)
{
    Unit u = Unit::parse("cm");
    EXPECT_TRUE(u.same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1.0e-2);
}

TEST(UnitTest, PHz)
{
    Unit u = Unit::parse("PHz");
    EXPECT_TRUE(u.same_dimension(Unit::parse("Hz")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1.0e15);
}

TEST(UnitTest, DB)
{
    Unit u = Unit::parse("dB");
    EXPECT_FALSE(u.has_dimension());
    EXPECT_DOUBLE_EQ(u.multiplier(), 1.0);
}

TEST(UnitTest, Nmi)
{
    Unit u = Unit::parse("nmi");
    EXPECT_TRUE(u.same_dimension(Unit::parse("meter")));
    EXPECT_DOUBLE_EQ(u.multiplier(), 1852);
}

// =========================================================================
//  Unit aliases
// =========================================================================

TEST(UnitTest, MeterAliases)
{
    EXPECT_TRUE(Unit::parse("meter").same_dimension(Unit::parse("meters")));
    EXPECT_TRUE(Unit::parse("metre").same_dimension(Unit::parse("metres")));
    EXPECT_DOUBLE_EQ(Unit::parse("meter").multiplier(), 1.0);
    EXPECT_DOUBLE_EQ(Unit::parse("meters").multiplier(), 1.0);
}

TEST(UnitTest, SecAlias)
{
    EXPECT_DOUBLE_EQ(Unit::parse("sec").multiplier(), 1.0);
}

TEST(UnitTest, OhmAliases)
{
    EXPECT_TRUE(Unit::parse("Ohm").same_dimension(Unit::parse("Ohms")));
    EXPECT_DOUBLE_EQ(Unit::parse("Ohm").multiplier(), 1.0);
    EXPECT_DOUBLE_EQ(Unit::parse("Ohms").multiplier(), 1.0);
}

// =========================================================================
//  Compound not supported
// =========================================================================

TEST(UnitTest, CompoundRejected)
{
    EXPECT_THROW(Unit::parse("MHz/kOhm"), std::invalid_argument);
}

// =========================================================================
//  Case sensitivity
// =========================================================================

TEST(UnitTest, CaseSensitivityMvsMilli)
{
    Unit uM = Unit::parse("MHz");
    Unit um = Unit::parse("mHz");
    EXPECT_DOUBLE_EQ(uM.multiplier(), 1e6);
    EXPECT_DOUBLE_EQ(um.multiplier(), 1e-3);
    EXPECT_TRUE(uM.same_dimension(um));
}

TEST(UnitTest, CaseSensitivityFvsFemto)
{
    Unit uF = Unit::parse("F");
    EXPECT_TRUE(uF.has_dimension());

    Unit ufF = Unit::parse("fF");
    EXPECT_TRUE(ufF.same_dimension(uF));
    EXPECT_DOUBLE_EQ(ufF.multiplier(), 1e-15);
}

TEST(UnitTest, CaseSensitivityAvsAtto)
{
    Unit uA = Unit::parse("A");
    EXPECT_TRUE(uA.has_dimension());

    Unit uaA = Unit::parse("aA");
    EXPECT_TRUE(uaA.same_dimension(uA));
    EXPECT_DOUBLE_EQ(uaA.multiplier(), 1e-18);
}

// =========================================================================
//  canonicalize
// =========================================================================

TEST(UnitTest, CanonicalizeStripsMultiplier)
{
    Unit cm = Unit::parse("cm");
    Unit c = cm.canonicalized();
    EXPECT_DOUBLE_EQ(c.multiplier(), 1.0);
    EXPECT_TRUE(c.same_dimension(cm));
}

TEST(UnitTest, CanonicalizeStripsMHz)
{
    Unit mhz = Unit::parse("MHz");
    Unit c = mhz.canonicalized();
    EXPECT_DOUBLE_EQ(c.multiplier(), 1.0);
    EXPECT_TRUE(c.same_dimension(mhz));
}

// =========================================================================
//  same_dimension
// =========================================================================

TEST(UnitTest, SameDimensionTrue)
{
    EXPECT_TRUE(Unit::parse("meter").same_dimension(Unit::parse("meter")));
    EXPECT_TRUE(Unit::parse("meter").same_dimension(Unit::parse("cm")));
    EXPECT_TRUE(Unit::parse("Hz").same_dimension(Unit::parse("Hz")));
    EXPECT_TRUE(Unit::parse("Hz").same_dimension(Unit::parse("MHz")));
}

TEST(UnitTest, SameDimensionFalse)
{
    EXPECT_FALSE(Unit::parse("meter").same_dimension(Unit::parse("sec")));
    EXPECT_FALSE(Unit::parse("Hz").same_dimension(Unit::parse("V")));
    EXPECT_FALSE(Unit::parse("A").same_dimension(Unit::parse("W")));
}

// =========================================================================
//  has_dimension / is_dimensionless
// =========================================================================

TEST(UnitTest, HasDimensionTrue)
{
    EXPECT_TRUE(Unit::parse("meter").has_dimension());
    EXPECT_TRUE(Unit::parse("Hz").has_dimension());
    EXPECT_TRUE(Unit::parse("V").has_dimension());
}

TEST(UnitTest, HasDimensionFalseForScaleFactors)
{
    EXPECT_FALSE(Unit::parse("M").has_dimension());
    EXPECT_FALSE(Unit::parse("k").has_dimension());
    EXPECT_FALSE(Unit::parse("m").has_dimension());
}

TEST(UnitTest, IsDimensionlessAndUnit)
{
    Unit dflt;
    EXPECT_TRUE(dflt.is_dimensionless());

    Unit underscore = Unit::parse("_");
    EXPECT_TRUE(underscore.is_dimensionless());

    Unit hz = Unit::parse("Hz");
    EXPECT_FALSE(hz.is_dimensionless());

    Unit meg = Unit::parse("M");
    EXPECT_FALSE(meg.is_dimensionless());
}

// =========================================================================
//  multiply_dim / divide_dim / pow_dim
// =========================================================================

TEST(UnitTest, MultiplyDim)
{
    Unit m = Unit::parse("meter").canonicalized();
    Unit pers = Unit::parse("Hz").canonicalized();
    Unit r = m*(pers);
    EXPECT_DOUBLE_EQ(r.multiplier(), 1.0);
    Unit ms = Unit::parse("meter").canonicalized()/(
        Unit::parse("sec").canonicalized());
    EXPECT_TRUE(r.same_dimension(ms));
}

TEST(UnitTest, DivideDim)
{
    Unit m = Unit::parse("meter").canonicalized();
    Unit s = Unit::parse("sec").canonicalized();
    Unit r = m/(s);
    EXPECT_DOUBLE_EQ(r.multiplier(), 1.0);
}


TEST(UnitTest, ToStringHz)
{
    Unit u = Unit::parse("Hz").canonicalized();
    EXPECT_FALSE(u.to_string().empty());
}

TEST(UnitTest, ToStringV)
{
    Unit u = Unit::parse("V").canonicalized();
    EXPECT_FALSE(u.to_string().empty());
}

TEST(UnitTest, ToStringA)
{
    Unit u = Unit::parse("A").canonicalized();
    EXPECT_FALSE(u.to_string().empty());
}

TEST(UnitTest, ToStringW)
{
    Unit u = Unit::parse("W").canonicalized();
    EXPECT_FALSE(u.to_string().empty());
}

TEST(UnitTest, ToStringOhm)
{
    Unit u = Unit::parse("Ohm").canonicalized();
    EXPECT_FALSE(u.to_string().empty());
}

TEST(UnitTest, ToStringMeter)
{
    Unit u = Unit::parse("meter").canonicalized();
    EXPECT_FALSE(u.to_string().empty());
}

TEST(UnitTest, ToStringSec)
{
    Unit u = Unit::parse("sec").canonicalized();
    EXPECT_FALSE(u.to_string().empty());
}

TEST(UnitTest, ToStringCompound)
{
    Unit m = Unit::parse("meter").canonicalized();
    Unit s = Unit::parse("sec").canonicalized();
    Unit ms = m/(s);
    EXPECT_FALSE(ms.to_string().empty());
}

TEST(UnitTest, ToStringDefaultDoesNotCrash)
{
    Unit u;
    SUCCEED();
}

// =========================================================================
//  equals / not-equals
// =========================================================================

TEST(UnitTest, Equals)
{
    Unit a = Unit::parse("Hz");
    Unit b = Unit::parse("Hz");
    EXPECT_TRUE(a == b);
}

TEST(UnitTest, NotEquals)
{
    Unit a = Unit::parse("Hz");
    Unit b = Unit::parse("V");
    EXPECT_TRUE(a != b);
}

TEST(UnitTest, NotEqualsMeterVsSec)
{
    Unit a = Unit::parse("meter");
    Unit b = Unit::parse("sec");
    EXPECT_TRUE(a != b);
}

// =========================================================================
//  Predef cannot stack with scale
// =========================================================================

TEST(UnitTest, PredefRejectsScale)
{
    EXPECT_THROW(Unit::parse("kin"), std::invalid_argument);
    EXPECT_THROW(Unit::parse("Mmil"), std::invalid_argument);
}

} // namespace xdataset
