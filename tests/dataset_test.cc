#include "dataset.h"
#include "block_fixtures.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace xdataset
{
    namespace
    {
        using namespace block_fixtures;

        // --------------------------------------------------------------------
        // Helpers
        // --------------------------------------------------------------------

        BlockCreateInfo makeBlockInfo(std::size_t indep_count = 2,
                                      std::size_t dep_count = 1)
        {
            BlockCreateInfo info;

            for (std::size_t i = 0; i < indep_count; ++i)
            {
                info.independent_specs.push_back(
                    IndependentSpec{"iv" + std::to_string(i),
                                    MakeScalarSeries(2),
                                    DimensionSpec::Regular(2)});
            }

            for (std::size_t j = 0; j < dep_count; ++j)
            {
                info.dependent_specs.push_back(
                    DependentSpec{"dv" + std::to_string(j),
                                  MakeScalarSeries(4)});
            }

            return info;
        }

    } // namespace

    // ========================================================================
    // Construction
    // ========================================================================

    TEST(DatasetTest, DefaultAndNamedConstruction)
    {
        Dataset ds;
        EXPECT_TRUE(ds.name().empty());

        Dataset named("noise");
        EXPECT_EQ(named.name(), "noise");

        named.set_name("other");
        EXPECT_EQ(named.name(), "other");
    }

    // ========================================================================
    // AddBlock
    // ========================================================================

    TEST(DatasetTest, AddBlockCreatesAnalysisAndBlock)
    {
        Dataset ds("noise");
        Block& b = ds.AddBlock("SP1", "SP", makeBlockInfo());

        EXPECT_EQ(b.name(), "noise.SP1.SP");
        EXPECT_TRUE(ds.HasAnalysis("SP1"));
        EXPECT_TRUE(ds.HasBlock("SP1", "SP"));
        EXPECT_EQ(ds.analysis_count(), 1u);
    }

    TEST(DatasetTest, AddBlockMoveOverload)
    {
        Dataset ds("noise");
        BlockCreateInfo info = makeBlockInfo();

        Block& b = ds.AddBlock("SP1", "SP", std::move(info));
        EXPECT_EQ(b.name(), "noise.SP1.SP");
    }

    TEST(DatasetTest, AddBlockDuplicateResultTypeThrows)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        EXPECT_THROW(
            { ds.AddBlock("SP1", "SP", makeBlockInfo()); },
            std::invalid_argument);
    }

    TEST(DatasetTest, AddBlockDifferentResultTypesSameAnalysis)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP1", "HB", makeBlockInfo());

        EXPECT_TRUE(ds.HasBlock("SP1", "SP"));
        EXPECT_TRUE(ds.HasBlock("SP1", "HB"));
        EXPECT_EQ(ds.analysis_count(), 1u);
    }

    TEST(DatasetTest, AddBlockDifferentAnalysesSameResultType)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP2", "SP", makeBlockInfo());

        EXPECT_TRUE(ds.HasBlock("SP1", "SP"));
        EXPECT_TRUE(ds.HasBlock("SP2", "SP"));
        EXPECT_EQ(ds.analysis_count(), 2u);
    }

    // ========================================================================
    // RemoveBlock / RemoveAnalysis
    // ========================================================================

    TEST(DatasetTest, RemoveBlockSucceeds)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP1", "HB", makeBlockInfo());

        std::size_t removed = ds.RemoveBlock("SP1", "SP");
        EXPECT_EQ(removed, 1u);
        EXPECT_FALSE(ds.HasBlock("SP1", "SP"));
        EXPECT_TRUE(ds.HasBlock("SP1", "HB"));
    }

    TEST(DatasetTest, RemoveBlockOnMissingAnalysisReturnsZero)
    {
        Dataset ds("noise");
        EXPECT_EQ(ds.RemoveBlock("Nope", "SP"), 0u);
    }

    TEST(DatasetTest, RemoveBlockOnMissingResultTypeReturnsZero)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        EXPECT_EQ(ds.RemoveBlock("SP1", "HB"), 0u);
    }

    TEST(DatasetTest, RemoveLastBlockCleansUpAnalysis)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        ds.RemoveBlock("SP1", "SP");
        EXPECT_FALSE(ds.HasAnalysis("SP1"));
        EXPECT_EQ(ds.analysis_count(), 0u);
    }

    TEST(DatasetTest, RemoveAnalysisClearsAllBlocks)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP1", "HB", makeBlockInfo());
        ds.AddBlock("SP2", "SP", makeBlockInfo());

        std::size_t removed = ds.RemoveAnalysis("SP1");
        EXPECT_EQ(removed, 1u);
        EXPECT_FALSE(ds.HasAnalysis("SP1"));
        EXPECT_TRUE(ds.HasAnalysis("SP2"));
        EXPECT_EQ(ds.analysis_count(), 1u);
    }

    TEST(DatasetTest, RemoveAnalysisOnMissingReturnsZero)
    {
        Dataset ds("noise");
        EXPECT_EQ(ds.RemoveAnalysis("Nope"), 0u);
    }

    // ========================================================================
    // HasAnalysis / HasBlock / HasUniqueDataArray
    // ========================================================================

    TEST(DatasetTest, HasQueriesCorrect)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        EXPECT_TRUE(ds.HasAnalysis("SP1"));
        EXPECT_FALSE(ds.HasAnalysis("SP2"));

        EXPECT_TRUE(ds.HasBlock("SP1", "SP"));
        EXPECT_FALSE(ds.HasBlock("SP1", "HB"));
    }

    TEST(DatasetTest, HasUniqueDataArrayTrue)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        EXPECT_TRUE(ds.HasUniqueDataArray("iv0"));
        EXPECT_TRUE(ds.HasUniqueDataArray("dv0"));
    }

    TEST(DatasetTest, HasUniqueDataArrayFalseWhenNone)
    {
        Dataset ds("noise");
        EXPECT_FALSE(ds.HasUniqueDataArray("nonexistent"));
    }

    TEST(DatasetTest, HasUniqueDataArrayFalseWhenMultiple)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP1", "HB", makeBlockInfo());

        // "iv0" appears in both SP and HB, so not unique
        EXPECT_FALSE(ds.HasUniqueDataArray("iv0"));
    }

    // ========================================================================
    // GetDataArray -- full path
    // ========================================================================

    TEST(DatasetTest, GetDataArrayFullPath)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        const DataArray& da = ds.GetDataArray("SP1", "SP", "iv0");
        EXPECT_EQ(da.data_kind(), DataArrayKind::kIndependent);
    }

    TEST(DatasetTest, GetDataArrayFullPathThrowsOnMissingAnalysis)
    {
        Dataset ds("noise");
        EXPECT_THROW(
            { ds.GetDataArray("Nope", "SP", "freq"); },
            std::out_of_range);
    }

    TEST(DatasetTest, GetDataArrayFullPathThrowsOnMissingBlock)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        EXPECT_THROW(
            { ds.GetDataArray("SP1", "HB", "freq"); },
            std::out_of_range);
    }

    TEST(DatasetTest, GetDataArrayFullPathThrowsOnMissingDataArray)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        EXPECT_THROW(
            { ds.GetDataArray("SP1", "SP", "nonexistent"); },
            std::invalid_argument);
    }

    // ========================================================================
    // GetDataArray -- unique name shortcut
    // ========================================================================

    TEST(DatasetTest, GetDataArrayUniqueNameShortcut)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        const DataArray& da = ds.GetDataArray("iv0");
    }

    TEST(DatasetTest, GetDataArrayUniqueNameThrowsOnNotFound)
    {
        Dataset ds("noise");
        EXPECT_THROW(
            { ds.GetDataArray("nonexistent"); },
            std::invalid_argument);
    }

    TEST(DatasetTest, GetDataArrayUniqueNameThrowsOnAmbiguous)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP1", "HB", makeBlockInfo());

        // "iv0" exists in both blocks
        EXPECT_THROW(
            { ds.GetDataArray("iv0"); },
            std::invalid_argument);
    }

    // ========================================================================
    // GetAnalysis / GetBlock -- const & mutable
    // ========================================================================

    TEST(DatasetTest, GetAnalysisConst)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP1", "HB", makeBlockInfo());

        const Dataset& cds = ds;
        const Dataset::ResultTypeMap& results = cds.GetAnalysis("SP1");
        EXPECT_EQ(results.size(), 2u);
    }

    TEST(DatasetTest, GetAnalysisThrowsOnMissing)
    {
        Dataset ds("noise");
        EXPECT_THROW({ ds.GetAnalysis("Nope"); }, std::out_of_range);
    }

    TEST(DatasetTest, GetBlockConst)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        const Dataset& cds = ds;
        const Block& b = cds.GetBlock("SP1", "SP");
        EXPECT_EQ(b.name(), "noise.SP1.SP");
    }

    TEST(DatasetTest, GetBlockThrowsOnMissing)
    {
        Dataset ds("noise");
        EXPECT_THROW({ ds.GetBlock("Nope", "SP"); }, std::out_of_range);
    }

    TEST(DatasetTest, GetBlockMutableAllowsLazyDataArrayCreation)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        Block& b = ds.GetBlock("SP1", "SP");
        const DataArray& da = b.GetOrCreateDataArray("iv0");
    }

    // ========================================================================
    // GetAnalysisNames / GetDataArrayNames
    // ========================================================================

    TEST(DatasetTest, GetAnalysisNamesEmpty)
    {
        Dataset ds("noise");
        EXPECT_TRUE(ds.GetAnalysisNames().empty());
    }

    TEST(DatasetTest, GetAnalysisNamesInsertionOrder)
    {
        Dataset ds("noise");
        ds.AddBlock("SP2", "SP", makeBlockInfo());
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP3", "SP", makeBlockInfo());

        auto names = ds.GetAnalysisNames();
        ASSERT_EQ(names.size(), 3u);
        EXPECT_EQ(names[0], "SP2");
        EXPECT_EQ(names[1], "SP1");
        EXPECT_EQ(names[2], "SP3");
    }

    TEST(DatasetTest, GetDataArrayNames)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());

        auto names = ds.GetDataArrayNames("SP1", "SP");
        ASSERT_EQ(names.size(), 3u);  // iv0, iv1, dv0
        EXPECT_EQ(names[0], "iv0");
        EXPECT_EQ(names[1], "iv1");
        EXPECT_EQ(names[2], "dv0");
    }

    TEST(DatasetTest, GetDataArrayNamesThrowsOnMissing)
    {
        Dataset ds("noise");
        EXPECT_THROW(
            { ds.GetDataArrayNames("Nope", "SP"); },
            std::invalid_argument);
    }

    // ========================================================================
    // analysis_count
    // ========================================================================

    TEST(DatasetTest, AnalysisCountEmpty)
    {
        Dataset ds("noise");
        EXPECT_EQ(ds.analysis_count(), 0u);
    }

    TEST(DatasetTest, AnalysisCountWithMultipleAnalyses)
    {
        Dataset ds("noise");
        ds.AddBlock("SP1", "SP", makeBlockInfo());
        ds.AddBlock("SP1", "HB", makeBlockInfo());
        ds.AddBlock("SP2", "SP", makeBlockInfo());

        EXPECT_EQ(ds.analysis_count(), 2u);
    }
}
