#include "hdf5_io.h"
#include "dataset_io.h"
#include "dataset.h"
#include "block.h"
#include "block_fixtures.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <stdexcept>

namespace xdataset
{
    namespace
    {
        using namespace block_fixtures;

        // --------------------------------------------------------------------
        // Helpers
        // --------------------------------------------------------------------

        BlockCreateInfo make_simple_info()
        {
            // R(2) × freq(3): 6 rows
            BlockCreateInfo info;
            info.independent_specs.push_back(
                IndependentSpec{"R", MakeScalarSeries(2),
                                DimensionSpec::Regular(2)});
            info.independent_specs.push_back(
                IndependentSpec{"freq", MakeScalarSeriesFrom({1.0, 2.0, 3.0}),
                                DimensionSpec::Regular(3)});
            info.dependent_specs.push_back(
                DependentSpec{"Vout", MakeScalarSeries(6)});
            return info;
        }

        BlockCreateInfo make_vector_info()
        {
            // Regular(2) → 2 rows, vector width 3
            BlockCreateInfo info;
            info.independent_specs.push_back(
                IndependentSpec{"x", MakeScalarSeries(2),
                                DimensionSpec::Regular(2)});
            info.dependent_specs.push_back(
                DependentSpec{"v", MakeVectorSeries(2, 3)});
            return info;
        }

    } // namespace

    // ========================================================================
    // Save / Load
    // ========================================================================

    TEST(Hdf5IoTest, SaveAndLoadSimpleRoundtrip)
    {
        Dataset ds("test_ds");
        ds.AddBlock("group/blk", make_simple_info());

        // Save
        DatasetIO::Save(ds, "hdf5", "test_roundtrip.h5");

        // Load
        Dataset loaded = DatasetIO::Load("hdf5", "test_roundtrip.h5");

        EXPECT_EQ(loaded.name(), "test_ds");
        EXPECT_EQ(loaded.block_count(), 1u);
        EXPECT_TRUE(loaded.HasBlock("group/blk"));

        const Block& b = loaded.GetBlock("group/blk");
        EXPECT_EQ(b.independents().size(), 2u);
        EXPECT_EQ(b.dependents().size(), 1u);

        // Verify independent data
        const IndependentSpec& r = b.independent_spec("R");
        EXPECT_EQ(r.data.size(), 2u);
        EXPECT_EQ(r.data.scalar_at<double>(0), 0.0);
        EXPECT_TRUE(r.dimension.is_regular());
        EXPECT_EQ(r.dimension.regular_size(), 2u);

        const IndependentSpec& freq = b.independent_spec("freq");
        EXPECT_EQ(freq.data.size(), 3u);
        EXPECT_DOUBLE_EQ(freq.data.scalar_at<double>(0), 1.0);
        EXPECT_DOUBLE_EQ(freq.data.scalar_at<double>(2), 3.0);
        EXPECT_TRUE(freq.dimension.is_regular());
        EXPECT_EQ(freq.dimension.regular_size(), 3u);

        // Verify dependent
        const DependentSpec& vout = b.dependent_spec("Vout");
        EXPECT_EQ(vout.data.size(), 6u);
    }

    TEST(Hdf5IoTest, SaveAndLoadNestedBlocks)
    {
        Dataset ds("nested");
        ds.AddBlock("simulation/SP1/SP", make_simple_info());
        ds.AddBlock("simulation/SP1/HB", make_simple_info());
        ds.AddBlock("summary/stats", make_vector_info());

        DatasetIO::Save(ds, "hdf5", "test_nested.h5");
        Dataset loaded = DatasetIO::Load("hdf5", "test_nested.h5");

        EXPECT_EQ(loaded.name(), "nested");
        EXPECT_EQ(loaded.block_count(), 3u);
        EXPECT_TRUE(loaded.HasBlock("simulation/SP1/SP"));
        EXPECT_TRUE(loaded.HasBlock("simulation/SP1/HB"));
        EXPECT_TRUE(loaded.HasBlock("summary/stats"));

        // Group structure is preserved
        EXPECT_TRUE(loaded.HasGroup("simulation"));
        EXPECT_TRUE(loaded.HasGroup("simulation/SP1"));
    }

    TEST(Hdf5IoTest, SaveAndLoadVectorDependent)
    {
        Dataset ds("vec_ds");
        ds.AddBlock("results", make_vector_info());

        DatasetIO::Save(ds, "hdf5", "test_vector.h5");
        Dataset loaded = DatasetIO::Load("hdf5", "test_vector.h5");

        const Block& b = loaded.GetBlock("results");
        const DependentSpec& v = b.dependent_spec("v");
        EXPECT_EQ(v.data.data_kind(), DataKind::kVector);
        EXPECT_EQ(v.data.data_shape().size(), 1u);
        EXPECT_EQ(v.data.data_shape()[0], 3);
        EXPECT_EQ(v.data.size(), 2u);
    }

    TEST(Hdf5IoTest, SaveAndLoadWithRaggedDimension)
    {
        Dataset ds("ragged");
        ds.AddBlock("data", MakeRaggedCreateInfo());

        DatasetIO::Save(ds, "hdf5", "test_ragged.h5");
        Dataset loaded = DatasetIO::Load("hdf5", "test_ragged.h5");

        EXPECT_EQ(loaded.block_count(), 1u);
        const Block& b = loaded.GetBlock("data");

        // Check ragged dimension
        const IndependentSpec& y = b.independent_spec("y");
        EXPECT_TRUE(y.dimension.is_ragged());
        EXPECT_EQ(y.dimension.ragged_sizes().size(), 2u);
        EXPECT_EQ(y.dimension.ragged_sizes()[0], 1u);
        EXPECT_EQ(y.dimension.ragged_sizes()[1], 2u);

        // Also check x is regular
        const IndependentSpec& x = b.independent_spec("x");
        EXPECT_TRUE(x.dimension.is_regular());
        EXPECT_EQ(x.dimension.regular_size(), 2u);
    }

    TEST(Hdf5IoTest, SaveAndGetDataArray) {
        Dataset ds("da_ds");
        ds.AddBlock("b", make_simple_info());

        DatasetIO::Save(ds, "hdf5", "test_da.h5");
        Dataset loaded = DatasetIO::Load("hdf5", "test_da.h5");

        const DataArray& da = loaded.GetDataArray("b", "freq");
        EXPECT_EQ(da.data_kind(), DataArrayKind::kIndependent);

        const DataArray& dep = loaded.GetDataArray("b", "Vout");
        EXPECT_EQ(dep.data_kind(), DataArrayKind::kDependent);
    }

    TEST(Hdf5IoTest, WriterReaderDirect) {
        Dataset ds("direct");
        ds.AddBlock("a", make_simple_info());

        {
            Hdf5Writer w("test_direct.h5");
            w.Write(ds);
        }

        Hdf5Reader r("test_direct.h5");
        Dataset loaded = r.Read();

        EXPECT_EQ(loaded.name(), "direct");
        EXPECT_EQ(loaded.block_count(), 1u);
        EXPECT_TRUE(loaded.HasBlock("a"));
    }

    TEST(Hdf5IoTest, UnsupportedFormatThrows)
    {
        Dataset ds("test");
        EXPECT_THROW({ DatasetIO::Save(ds, "json", "test.json"); },
                     std::invalid_argument);
    }

} // namespace xdataset
