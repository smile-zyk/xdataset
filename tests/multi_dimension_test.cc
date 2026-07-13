#include "multi_dimension_spec.h"

#include <gtest/gtest.h>

namespace xdataset
{
    TEST(MultiDimensionSpecTest, TracksRankAndCellCount)
    {
        MultiDimensionSpec spec;
        EXPECT_TRUE(spec.empty());

        spec.add_uniform(2).add_uniform(3);

        EXPECT_FALSE(spec.empty());
        EXPECT_EQ(spec.rank(), 2u);
        EXPECT_EQ(spec.ndim(), 2u);
        ASSERT_EQ(spec.dims().size(), 2u);
        EXPECT_TRUE(spec.dims()[0].is_uniform());
        EXPECT_TRUE(spec.dims()[1].is_uniform());
        EXPECT_EQ(spec.compute_cell_count(), 6u);
    }

    TEST(MultiDimensionSpecTest, ForEachLeafRowReportsDimensionRowIndices)
    {
        MultiDimensionSpec spec;
        spec.add_uniform(2).add_jagged({2, 1});

        std::vector<std::vector<std::size_t>> multi_indices;
        std::vector<std::vector<std::size_t>> dimension_row_indices;
        std::vector<std::size_t> row_flats;

        spec.for_each_leaf_row(
            [&](const MultiDimensionSpec::LeafRow& leaf_row)
            {
                multi_indices.push_back(leaf_row.multi_index);
                dimension_row_indices.push_back(leaf_row.dimension_row_indices);
                row_flats.push_back(leaf_row.row_flat);
            });

        ASSERT_EQ(multi_indices.size(), 3u);
        ASSERT_EQ(dimension_row_indices.size(), 3u);
        ASSERT_EQ(row_flats.size(), 3u);

        EXPECT_EQ(multi_indices[0], std::vector<std::size_t>({0, 0}));
        EXPECT_EQ(dimension_row_indices[0], std::vector<std::size_t>({0, 0}));
        EXPECT_EQ(row_flats[0], 0u);

        EXPECT_EQ(multi_indices[1], std::vector<std::size_t>({0, 1}));
        EXPECT_EQ(dimension_row_indices[1], std::vector<std::size_t>({0, 1}));
        EXPECT_EQ(row_flats[1], 1u);

        EXPECT_EQ(multi_indices[2], std::vector<std::size_t>({1, 0}));
        EXPECT_EQ(dimension_row_indices[2], std::vector<std::size_t>({1, 2}));
        EXPECT_EQ(row_flats[2], 2u);
    }

    TEST(MultiDimensionSpecTest, ForEachLeafRowHandlesInterleavedUniformAndJaggedDimensions)
    {
        MultiDimensionSpec spec;
        spec.add_uniform(2).add_jagged({2, 1}).add_uniform(2);

        std::vector<std::vector<std::size_t>> multi_indices;
        std::vector<std::vector<std::size_t>> dimension_row_indices;
        std::vector<std::size_t> row_flats;

        spec.for_each_leaf_row(
            [&](const MultiDimensionSpec::LeafRow& leaf_row)
            {
                multi_indices.push_back(leaf_row.multi_index);
                dimension_row_indices.push_back(leaf_row.dimension_row_indices);
                row_flats.push_back(leaf_row.row_flat);
            });

        ASSERT_EQ(multi_indices.size(), 6u);
        ASSERT_EQ(dimension_row_indices.size(), 6u);
        ASSERT_EQ(row_flats.size(), 6u);

        EXPECT_EQ(multi_indices[0], std::vector<std::size_t>({0, 0, 0}));
        EXPECT_EQ(dimension_row_indices[0], std::vector<std::size_t>({0, 0, 0}));
        EXPECT_EQ(row_flats[0], 0u);

        EXPECT_EQ(multi_indices[1], std::vector<std::size_t>({0, 0, 1}));
        EXPECT_EQ(dimension_row_indices[1], std::vector<std::size_t>({0, 0, 1}));
        EXPECT_EQ(row_flats[1], 1u);

        EXPECT_EQ(multi_indices[2], std::vector<std::size_t>({0, 1, 0}));
        EXPECT_EQ(dimension_row_indices[2], std::vector<std::size_t>({0, 1, 0}));
        EXPECT_EQ(row_flats[2], 2u);

        EXPECT_EQ(multi_indices[3], std::vector<std::size_t>({0, 1, 1}));
        EXPECT_EQ(dimension_row_indices[3], std::vector<std::size_t>({0, 1, 1}));
        EXPECT_EQ(row_flats[3], 3u);

        EXPECT_EQ(multi_indices[4], std::vector<std::size_t>({1, 0, 0}));
        EXPECT_EQ(dimension_row_indices[4], std::vector<std::size_t>({1, 2, 0}));
        EXPECT_EQ(row_flats[4], 4u);

        EXPECT_EQ(multi_indices[5], std::vector<std::size_t>({1, 0, 1}));
        EXPECT_EQ(dimension_row_indices[5], std::vector<std::size_t>({1, 2, 1}));
        EXPECT_EQ(row_flats[5], 5u);
    }

    TEST(MultiDimensionSpecTest, ProjectSelectionKeepsAndDropsDimensions)
    {
        MultiDimensionSpec spec;
        spec.add_uniform(2).add_uniform(3).add_uniform(4);

        std::vector<MultiIndexSelector> selectors;
        selectors.push_back(MultiIndexSelector::Any());
        selectors.push_back(MultiIndexSelector::Equal(1));
        selectors.push_back(MultiIndexSelector::In({0, 2}));

        const MultiDimensionSpec::SelectionProjection projection =
            spec.project_selection(selectors);
        const MultiDimensionSpec selected(projection.projected_dims);

        EXPECT_EQ(selected.rank(), 2u);
        ASSERT_EQ(selected.dims().size(), 2u);
        EXPECT_TRUE(selected.dims()[0].is_uniform());
        EXPECT_TRUE(selected.dims()[1].is_uniform());
        EXPECT_EQ(selected.dims()[0].uniform_size(), 2);
        EXPECT_EQ(selected.dims()[1].uniform_size(), 2);
    }
} // namespace xdataset