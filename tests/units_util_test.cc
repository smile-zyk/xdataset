#include "units_util.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace xdataset
{
    // -----------------------------------------------------------------------
    // parse_unit
    // -----------------------------------------------------------------------

    TEST(UnitsUtilTest, ParseValidUnit)
    {
        EXPECT_NO_THROW(parse_unit("m"));
        EXPECT_NO_THROW(parse_unit("m/s"));
        EXPECT_NO_THROW(parse_unit("Hz"));
        EXPECT_NO_THROW(parse_unit("kg*m^-2"));
    }

    TEST(UnitsUtilTest, ParseInvalidUnitThrows)
    {
        EXPECT_THROW(parse_unit("blarghzzz"), std::invalid_argument);
    }

    // -----------------------------------------------------------------------
    // is_affine
    // -----------------------------------------------------------------------

    TEST(UnitsUtilTest, IsAffineDegC)
    {
        EXPECT_TRUE(is_affine(parse_unit("degC")));
    }

    TEST(UnitsUtilTest, IsAffineDegF)
    {
        EXPECT_TRUE(is_affine(parse_unit("degF")));
    }

    TEST(UnitsUtilTest, IsAffineFalseForNormalUnits)
    {
        EXPECT_FALSE(is_affine(parse_unit("K")));
        EXPECT_FALSE(is_affine(parse_unit("m")));
        EXPECT_FALSE(is_affine(parse_unit("Hz")));
        EXPECT_FALSE(is_affine(parse_unit("Pa")));
    }

    // -----------------------------------------------------------------------
    // same_dimension
    // -----------------------------------------------------------------------

    TEST(UnitsUtilTest, SameDimensionTrue)
    {
        EXPECT_TRUE(same_dimension(parse_unit("m"), parse_unit("m")));
        EXPECT_TRUE(same_dimension(parse_unit("m"), parse_unit("km")));
        EXPECT_TRUE(same_dimension(parse_unit("Hz"), parse_unit("Hz")));
        EXPECT_TRUE(same_dimension(parse_unit("Pa"), parse_unit("psi")));
        // affine vs non-affine
        EXPECT_TRUE(same_dimension(parse_unit("degC"), parse_unit("K")));
        EXPECT_TRUE(same_dimension(parse_unit("degF"), parse_unit("K")));
    }

    TEST(UnitsUtilTest, SameDimensionFalse)
    {
        EXPECT_FALSE(same_dimension(parse_unit("m"), parse_unit("s")));
        EXPECT_FALSE(same_dimension(parse_unit("Hz"), parse_unit("Pa")));
        EXPECT_FALSE(same_dimension(parse_unit("N"), parse_unit("J")));
        EXPECT_FALSE(same_dimension(parse_unit("m"), parse_unit("degC")));
        EXPECT_FALSE(same_dimension(parse_unit("s"), parse_unit("degF")));
    }

    // -----------------------------------------------------------------------
    // multiplier_of
    // -----------------------------------------------------------------------

    TEST(UnitsUtilTest, MultiplierOf)
    {
        // Coherent SI units have multiplier 1
        EXPECT_DOUBLE_EQ(multiplier_of(parse_unit("Hz")), 1.0);
        EXPECT_DOUBLE_EQ(multiplier_of(parse_unit("Pa")), 1.0);
        EXPECT_DOUBLE_EQ(multiplier_of(parse_unit("N")), 1.0);

        // Prefixed / non-coherent
        EXPECT_DOUBLE_EQ(multiplier_of(parse_unit("km")), 1000.0);
        EXPECT_DOUBLE_EQ(multiplier_of(parse_unit("mm")), 0.001);
        EXPECT_DOUBLE_EQ(multiplier_of(parse_unit("GHz")), 1e9);

        // Non-SI
        EXPECT_NEAR(multiplier_of(parse_unit("psi")), 6894.76, 0.01);
        EXPECT_NEAR(multiplier_of(parse_unit("ft")), 0.3048, 1e-6);
        EXPECT_DOUBLE_EQ(multiplier_of(parse_unit("in")), 0.0254);
    }

    // -----------------------------------------------------------------------
    // canonicalize
    // -----------------------------------------------------------------------

    TEST(UnitsUtilTest, CanonicalizeStripsMultiplier)
    {
        Unit km = parse_unit("km");
        Unit c = canonicalize(km);
        EXPECT_DOUBLE_EQ(c.multiplier(), 1.0);
        EXPECT_TRUE(same_dimension(c, km));
    }

    TEST(UnitsUtilTest, CanonicalizeClearsEfflag)
    {
        Unit degC = parse_unit("degC");
        Unit c = canonicalize(degC);
        EXPECT_FALSE(is_affine(c));
        EXPECT_TRUE(same_dimension(c, degC));
    }

    // -----------------------------------------------------------------------
    // multiply_dim / divide_dim / pow_dim
    // -----------------------------------------------------------------------

    TEST(UnitsUtilTest, MultiplyDim)
    {
        Unit m = canonicalize(parse_unit("m"));
        Unit per_s = canonicalize(parse_unit("Hz"));  // s^-1
        Unit result = multiply_dim(m, per_s);
        EXPECT_TRUE(same_dimension(result, parse_unit("m/s")));
        EXPECT_DOUBLE_EQ(result.multiplier(), 1.0);
    }

    TEST(UnitsUtilTest, DivideDim)
    {
        Unit m = canonicalize(parse_unit("m"));
        Unit s = canonicalize(parse_unit("s"));
        Unit result = divide_dim(m, s);
        EXPECT_TRUE(same_dimension(result, parse_unit("m/s")));
        EXPECT_DOUBLE_EQ(result.multiplier(), 1.0);
    }

    TEST(UnitsUtilTest, PowDim)
    {
        Unit m = canonicalize(parse_unit("m"));
        Unit m2 = pow_dim(m, 2);
        EXPECT_TRUE(same_dimension(m2, canonicalize(parse_unit("m^2"))));
        EXPECT_DOUBLE_EQ(m2.multiplier(), 1.0);

        Unit m_inv = pow_dim(m, -1);
        EXPECT_TRUE(same_dimension(m_inv, canonicalize(parse_unit("1/m"))));
        EXPECT_DOUBLE_EQ(m_inv.multiplier(), 1.0);
    }

    // -----------------------------------------------------------------------
    // unit_string
    // -----------------------------------------------------------------------

    TEST(UnitsUtilTest, UnitStringRecoversSIDerivedNames)
    {
        EXPECT_EQ(unit_string(canonicalize(parse_unit("Hz"))), "Hz");
        EXPECT_EQ(unit_string(canonicalize(parse_unit("Pa"))), "Pa");
        EXPECT_EQ(unit_string(canonicalize(parse_unit("N"))), "N");
        EXPECT_EQ(unit_string(canonicalize(parse_unit("J"))), "J");
        EXPECT_EQ(unit_string(canonicalize(parse_unit("W"))), "W");
        EXPECT_EQ(unit_string(canonicalize(parse_unit("V"))), "V");
    }

    TEST(UnitsUtilTest, UnitStringComposite)
    {
        // m/s is recognised as-is
        EXPECT_EQ(unit_string(canonicalize(parse_unit("m/s"))), "m/s");
    }

    TEST(UnitsUtilTest, UnitStringNonCoherentFallsBack)
    {
        // psi/atm/km canonicalize to base-unit combinations
        Unit psi = canonicalize(parse_unit("psi"));
        EXPECT_NE(unit_string(psi), "psi");        // multiplier stripped, falls back
        EXPECT_FALSE(unit_string(psi).empty());
    }
} // namespace xdataset
