#include "../variable_data.h"

#include <gtest/gtest.h>

#include <vector>

using xdataset::CellSeries;
using xdataset::Index;
using xdataset::MultiDimensionSpec;
using xdataset::MultiIndexSelector;
using xdataset::VariableData;

TEST(MultiIndexSelectorTest, AnyAndEqual) {
    MultiIndexSelector any = MultiIndexSelector::Any();
    MultiIndexSelector exact = MultiIndexSelector::Equal(3);

    EXPECT_TRUE(any.matches(0));
    EXPECT_TRUE(any.matches(9));
    EXPECT_TRUE(exact.matches(3));
    EXPECT_FALSE(exact.matches(2));
}

TEST(MultiIndexSelectorTest, EqualMinusOneMeansAny) {
    MultiIndexSelector s = MultiIndexSelector::Equal(-1);
    EXPECT_TRUE(s.matches(0));
    EXPECT_TRUE(s.matches(5));
}

TEST(MultiIndexSelectorTest, InWithMinusOneMeansAny) {
    MultiIndexSelector s = MultiIndexSelector::In(std::vector<Index>{1, -1, 3});
    EXPECT_TRUE(s.matches(0));
    EXPECT_TRUE(s.matches(3));
}

TEST(MultiIndexSelectorTest, EqualAndInNewNames) {
    MultiIndexSelector equal = MultiIndexSelector::Equal(2);
    MultiIndexSelector in = MultiIndexSelector::In(std::vector<Index>{1, 2, 2, 3});

    EXPECT_TRUE(equal.is_equal());
    EXPECT_EQ(equal.equal_value(), 2);
    EXPECT_TRUE(equal.matches(2));
    EXPECT_FALSE(equal.matches(3));

    EXPECT_TRUE(in.is_in());
    EXPECT_TRUE(in.matches(1));
    EXPECT_TRUE(in.matches(2));
    EXPECT_FALSE(in.matches(4));
}

TEST(MultiDimensionSpecTest, MixedLevelsAndTotalCount) {
    // [3, 2, [2,3,6,8,1,2], 3]
    MultiDimensionSpec spec;
    spec.add_uniform(3)
        .add_uniform(2)
        .add_jagged(std::vector<Index>{2, 3, 6, 8, 1, 2})
        .add_uniform(3);

    std::vector<std::size_t> levels = spec.level_node_counts();
    ASSERT_EQ(levels.size(), 5u);
    EXPECT_EQ(levels[0], 1u);
    EXPECT_EQ(levels[1], 3u);
    EXPECT_EQ(levels[2], 6u);
    EXPECT_EQ(levels[3], 22u);
    EXPECT_EQ(levels[4], 66u);
    EXPECT_EQ(spec.total_cell_count(), 66u);
}

TEST(MultiDimensionSpecTest, ExactMultiToFlatAndBack) {
    MultiDimensionSpec spec;
    spec.add_uniform(2)
        .add_jagged(std::vector<Index>{3, 5});

    EXPECT_EQ(spec.flat_index(std::vector<Index>{0, 0}), 0u);
    EXPECT_EQ(spec.flat_index(std::vector<Index>{0, 2}), 2u);
    EXPECT_EQ(spec.flat_index(std::vector<Index>{1, 0}), 3u);
    EXPECT_EQ(spec.flat_index(std::vector<Index>{1, 4}), 7u);

    EXPECT_EQ(spec.multi_index(0), (std::vector<Index>{0, 0}));
    EXPECT_EQ(spec.multi_index(2), (std::vector<Index>{0, 2}));
    EXPECT_EQ(spec.multi_index(3), (std::vector<Index>{1, 0}));
    EXPECT_EQ(spec.multi_index(7), (std::vector<Index>{1, 4}));
}

TEST(MultiDimensionSpecTest, FlatToMultiOutOfRangeThrows) {
    MultiDimensionSpec spec;
    spec.add_uniform(2).add_jagged(std::vector<Index>{3, 5});
    EXPECT_THROW(spec.multi_index(8), std::out_of_range);
}

TEST(MultiDimensionSpecTest, QueryWithWildcardAndIn) {
    // Shape: [2, 3, 4] => flat = i*12 + j*4 + k
    MultiDimensionSpec spec;
    spec.add_uniform(2).add_uniform(3).add_uniform(4);

    std::vector<MultiIndexSelector> query;
    query.push_back(MultiIndexSelector::Any());
    query.push_back(MultiIndexSelector::In(std::vector<Index>{1, 2}));
    query.push_back(MultiIndexSelector::Equal(3));

    // Expected: (0,1,3)=7, (0,2,3)=11, (1,1,3)=19, (1,2,3)=23
    std::vector<std::size_t> flat = spec.flat_indices(query);
    ASSERT_EQ(flat.size(), 4u);
    EXPECT_EQ(flat[0], 7u);
    EXPECT_EQ(flat[1], 11u);
    EXPECT_EQ(flat[2], 19u);
    EXPECT_EQ(flat[3], 23u);
}

TEST(MultiDimensionSpecTest, QueryWithMinusOneConvenience) {
    // [-1, 2, 1] for shape [2,3,4]
    MultiDimensionSpec spec;
    spec.add_uniform(2).add_uniform(3).add_uniform(4);

    std::vector<std::size_t> flat = spec.flat_indices(std::vector<Index>{-1, 2, 1});
    ASSERT_EQ(flat.size(), 2u);
    EXPECT_EQ(flat[0], 9u);   // (0,2,1)
    EXPECT_EQ(flat[1], 21u);  // (1,2,1)
}

TEST(MultiDimensionSpecTest, InnermostGroupsUniformShape) {
    MultiDimensionSpec spec;
    spec.add_uniform(2).add_uniform(3).add_uniform(4);

    const std::vector<std::size_t> levels = spec.level_node_counts();
    ASSERT_EQ(levels[levels.size() - 2], 6u);

    const std::vector<std::vector<std::size_t> > groups = spec.innermost_groups();
    ASSERT_EQ(groups.size(), levels[levels.size() - 2]);

    EXPECT_EQ(groups[0], (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_EQ(groups[1], (std::vector<std::size_t>{4, 5, 6, 7}));
    EXPECT_EQ(groups[5], (std::vector<std::size_t>{20, 21, 22, 23}));
}

TEST(MultiDimensionSpecTest, InnermostGroupsJaggedShape) {
    // [2, [2,1], 2, [1,2,3,4,5,6]] total 21
    MultiDimensionSpec spec;
    spec.add_uniform(2)
        .add_jagged(std::vector<Index>{2, 1})
        .add_uniform(2)
        .add_jagged(std::vector<Index>{1, 2, 3, 4, 5, 6});

    const std::vector<std::size_t> levels = spec.level_node_counts();
    ASSERT_EQ(levels[levels.size() - 2], 6u);

    const std::vector<std::vector<std::size_t> > groups = spec.innermost_groups();
    ASSERT_EQ(groups.size(), levels[levels.size() - 2]);

    EXPECT_EQ(groups[0], (std::vector<std::size_t>{0}));
    EXPECT_EQ(groups[1], (std::vector<std::size_t>{1, 2}));
    EXPECT_EQ(groups[2], (std::vector<std::size_t>{3, 4, 5}));
    EXPECT_EQ(groups[3], (std::vector<std::size_t>{6, 7, 8, 9}));
    EXPECT_EQ(groups[4], (std::vector<std::size_t>{10, 11, 12, 13, 14}));
    EXPECT_EQ(groups[5], (std::vector<std::size_t>{15, 16, 17, 18, 19, 20}));
}

TEST(MultiDimensionSpecTest, ErgonomicApiAliasesWork) {
    MultiDimensionSpec spec = MultiDimensionSpec::UniformShape(std::vector<Index>{2, 3, 4});

    EXPECT_EQ(spec.ndim(), 3u);
    EXPECT_FALSE(spec.empty());
    EXPECT_EQ(spec.total_size(), 24u);
    EXPECT_EQ(spec.at(std::vector<Index>{1, 2, 3}), 23u);

    std::vector<std::size_t> selected =
        spec.select_flat_indices(std::vector<Index>{-1, 1, 2});
    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0], 6u);
    EXPECT_EQ(selected[1], 18u);
}

TEST(MultiDimensionSpecTest, QueryAgainstJaggedShape) {
    // [2, [2,1], 2, [1,2,3,4,5,6]] total 21
    MultiDimensionSpec spec;
    spec.add_uniform(2)
        .add_jagged(std::vector<Index>{2, 1})
        .add_uniform(2)
        .add_jagged(std::vector<Index>{1, 2, 3, 4, 5, 6});

    std::vector<MultiIndexSelector> query;
    query.push_back(MultiIndexSelector::Equal(1));
    query.push_back(MultiIndexSelector::Equal(0));
    query.push_back(MultiIndexSelector::Any());
    query.push_back(MultiIndexSelector::Equal(4));

    // Matches: (1,0,0,4)=14 and (1,0,1,4)=19
    std::vector<std::size_t> flat = spec.flat_indices(query);
    ASSERT_EQ(flat.size(), 2u);
    EXPECT_EQ(flat[0], 14u);
    EXPECT_EQ(flat[1], 19u);
}

TEST(MultiDimensionSpecTest, JaggedParentCountMismatchThrows) {
    MultiDimensionSpec spec;
    spec.add_uniform(3).add_jagged(std::vector<Index>{2, 4});

    EXPECT_THROW(spec.validate_definition(), std::invalid_argument);
    EXPECT_THROW(spec.total_cell_count(), std::invalid_argument);
    EXPECT_THROW(spec.build_layout(), std::invalid_argument);
}

TEST(VariableDataTest, WrapperDelegatesMappingAndSeries) {
    MultiDimensionSpec spec;
    spec.add_uniform(2).add_uniform(3).add_uniform(4);  // total 24

    CellSeries s = CellSeries::Scalars<int>(24, 0);
    VariableData vd(spec, s);

    EXPECT_EQ(vd.size(), 24u);

    std::size_t flat = vd.flat_index(std::vector<Index>{1, 2, 3});
    EXPECT_EQ(flat, 23u);

    vd.series().scalar_at<int>(flat) = 99;
    EXPECT_EQ(vd.series().scalar_at<int>(23), 99);
    EXPECT_EQ(vd.multi_index(23), (std::vector<Index>{1, 2, 3}));

    std::vector<std::size_t> selected = vd.flat_indices(std::vector<Index>{-1, 1, 2});
    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0], 6u);
    EXPECT_EQ(selected[1], 18u);
}

TEST(VariableDataTest, SelectRebuildsSeriesAndProjectedShape) {
    MultiDimensionSpec spec;
    spec.add_uniform(2).add_uniform(3).add_uniform(4);  // total 24

    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23});
    VariableData vd(spec, s);

    std::vector<MultiIndexSelector> query;
    query.push_back(MultiIndexSelector::Equal(1));
    query.push_back(MultiIndexSelector::Any());
    query.push_back(MultiIndexSelector::Equal(2));

    VariableData selected = vd.select(query);

    EXPECT_EQ(selected.size(), 3u);
    EXPECT_EQ(selected.multi_dimension_spec().rank(), 1u);
    EXPECT_EQ(selected.multi_dimension_spec().total_cell_count(), 3u);
    EXPECT_EQ(selected.multi_index(0), (std::vector<Index>{0}));
    EXPECT_EQ(selected.multi_index(2), (std::vector<Index>{2}));
    EXPECT_EQ(selected.series().scalar_at<int>(0), 14);
    EXPECT_EQ(selected.series().scalar_at<int>(1), 18);
    EXPECT_EQ(selected.series().scalar_at<int>(2), 22);
}

TEST(VariableDataTest, WrapperShapeMismatchThrows) {
    MultiDimensionSpec spec;
    spec.add_uniform(2).add_uniform(2);  // total 4

    CellSeries bad = CellSeries::Scalars<int>(3, 0);
    EXPECT_THROW(VariableData(spec, bad), std::invalid_argument);
}
