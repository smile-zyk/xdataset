#include "xdataset_predefine.h"

#include <gtest/gtest.h>

#include <string>

namespace xdataset
{

// =========================================================================
// ShortHash
// =========================================================================

TEST(ShortHashTest, DeterministicAndLowercaseHex)
{
    EXPECT_EQ(ShortHash(""), "1c9dc5");
    EXPECT_EQ(ShortHash("abc"), ShortHash("abc"));  // stable across calls
    EXPECT_EQ(ShortHash("hello world"), ShortHash("hello world"));

    // Six lowercase hex digits.
    const std::string h = ShortHash("xdataset");
    ASSERT_EQ(h.size(), 6u);
    for (char c : h)
    {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }
}

// =========================================================================
// ConvertToValidIdentifier
// =========================================================================

TEST(ConvertToValidIdentifierTest, AlreadyValidReturnsUnchanged)
{
    EXPECT_EQ(ConvertToValidIdentifier("abc"), "abc");
    EXPECT_EQ(ConvertToValidIdentifier("a1_b2"), "a1_b2");
    EXPECT_EQ(ConvertToValidIdentifier("_x"), "_x");
    EXPECT_EQ(ConvertToValidIdentifier("X"), "X");
}

TEST(ConvertToValidIdentifierTest, IllegalRunsCondensedToUnderscore)
{
    EXPECT_EQ(ConvertToValidIdentifier("a b"), "a_b");
    EXPECT_EQ(ConvertToValidIdentifier("a-b"), "a_b");
    EXPECT_EQ(ConvertToValidIdentifier("a!!b"), "a_b");
    EXPECT_EQ(ConvertToValidIdentifier("freq (GHz)"), "freq_GHz_");
    EXPECT_EQ(ConvertToValidIdentifier("my-block set"), "my_block_set");
}

TEST(ConvertToValidIdentifierTest, LeadingDigitGetsPrefix)
{
    EXPECT_EQ(ConvertToValidIdentifier("123abc"), "_123abc");
    EXPECT_EQ(ConvertToValidIdentifier("1x"), "_1x");
}

TEST(ConvertToValidIdentifierTest, EmptyAndAllIllegalFallBackToHash)
{
    const std::string a = ConvertToValidIdentifier("");
    const std::string b = ConvertToValidIdentifier("!!!");
    const std::string c = ConvertToValidIdentifier("   ");

    // Deterministic, prefixed, and unique for different inputs.
    EXPECT_EQ(a, ConvertToValidIdentifier(""));
    EXPECT_EQ(b, ConvertToValidIdentifier("!!!"));
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);

    EXPECT_EQ(a, "invalid_node_" + ShortHash(""));
    EXPECT_EQ(b, "invalid_node_" + ShortHash("!!!"));

    for (const std::string& s : {a, b, c})
    {
        EXPECT_TRUE(IsValidIdentifier(s));
    }
}

TEST(ConvertToValidIdentifierTest, AlwaysProducesValidIdentifier)
{
    const char* cases[] = {"", " ", "a b", "1st", "hello-world", "a/b",
                           "(x)", "a\tb", "a\n", "123", "-", "a.b.c"};
    for (const char* in : cases)
    {
        EXPECT_TRUE(IsValidIdentifier(ConvertToValidIdentifier(in)))
            << "input: " << in;
    }
}

} // namespace xdataset