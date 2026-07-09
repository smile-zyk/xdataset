#ifndef BLOCK_TEST_HELPERS_H
#define BLOCK_TEST_HELPERS_H

#include "block.h"

namespace xdataset
{
    namespace test_helpers
    {
        inline CellSeries MakeScalarSeries(std::size_t size)
        {
            return CellSeries::Scalars<double>(size, 0.0);
        }

        inline CellSeries MakeScalarSeriesFrom(const std::vector<double>& values)
        {
            return CellSeries::ScalarsFrom<double>(values);
        }

        inline BlockCreateInfo MakeBaseCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo";
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"x", MakeScalarSeries(2), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"y", MakeScalarSeries(3), DimensionSpec::Uniform(3)});
            info.dependent_variables.push_back(
                DependentVariableCreateInfo{"z", MakeScalarSeries(6)});
            return info;
        }

        inline BlockCreateInfo MakeValueRichCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-values";
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"y", MakeScalarSeriesFrom({1.0, 2.0, 3.0}), DimensionSpec::Uniform(3)});
            info.dependent_variables.push_back(
                DependentVariableCreateInfo{"z", MakeScalarSeriesFrom({100.0, 101.0, 102.0, 103.0, 104.0, 105.0})});
            return info;
        }

        inline BlockCreateInfo MakeJaggedCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-jagged";
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"y", MakeScalarSeriesFrom({1.0, 2.0, 3.0}), DimensionSpec::Jagged({1, 2})});
            info.dependent_variables.push_back(
                DependentVariableCreateInfo{"z", MakeScalarSeriesFrom({100.0, 101.0, 102.0})});
            return info;
        }

        inline BlockCreateInfo MakeInterleavedCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-interleaved";
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"y", MakeScalarSeriesFrom({1.0, 2.0, 3.0}), DimensionSpec::Jagged({1, 2})});
            info.independent_variables.push_back(
                IndependentVariableCreateInfo{"z", MakeScalarSeriesFrom({100.0, 200.0}), DimensionSpec::Uniform(2)});
            info.dependent_variables.push_back(
                DependentVariableCreateInfo{"w", MakeScalarSeriesFrom({1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0})});
            return info;
        }
    } // namespace test_helpers
} // namespace xdataset

#endif // BLOCK_TEST_HELPERS_H