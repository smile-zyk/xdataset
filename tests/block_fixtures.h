#ifndef BLOCK_FIXTURES_H
#define BLOCK_FIXTURES_H

#include "block.h"

namespace xdataset
{
    namespace block_fixtures
    {
        // --- low-level helpers --------------------------------------------------

        inline CellSeries MakeScalarSeries(std::size_t size)
        {
            return CellSeries::Scalars<double>(size, 0.0);
        }

        inline CellSeries MakeScalarSeriesFrom(const std::vector<double>& values)
        {
            return CellSeries::ScalarsFrom<double>(values);
        }

        inline CellSeries MakeIntScalarSeriesFrom(const std::vector<int>& values)
        {
            return CellSeries::ScalarsFrom<int>(values);
        }

        inline CellSeries MakeStringScalarSeriesFrom(const std::vector<std::string>& values)
        {
            CellSeries s = CellSeries::Scalars<std::string>(values.size());
            for (std::size_t i = 0; i < values.size(); ++i)
                s.scalar_at<std::string>(i) = values[i];
            return s;
        }

        inline CellSeries MakeVectorSeries(std::size_t rows, Index width)
        {
            CellSeries vecs = CellSeries::Vectors<double>(rows, width);
            for (std::size_t i = 0; i < rows; ++i)
            {
                Eigen::VectorXd v(width);
                for (Index j = 0; j < width; ++j)
                    v(j) = static_cast<double>(i * width + j + 1);
                vecs.vector_at<double>(i) = v;
            }
            return vecs;
        }

        inline CellSeries MakeMatrixSeries(std::size_t rows, Index r, Index c)
        {
            CellSeries mats = CellSeries::Matrices<double>(rows, r, c);
            for (std::size_t i = 0; i < rows; ++i)
            {
                Eigen::MatrixXd m(r, c);
                for (Index ri = 0; ri < r; ++ri)
                    for (Index ci = 0; ci < c; ++ci)
                        m(ri, ci) = static_cast<double>(i * r * c + ri * c + ci + 1);
                mats.matrix_at<double>(i) = m;
            }
            return mats;
        }

        // =====================================================================
        // Fixtures: uniform  (2 × 3)
        // =====================================================================

        inline BlockCreateInfo MakeBaseCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo";
            info.independent_variables.push_back(
                IndependentVariableInfo{"x", MakeScalarSeries(2), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"y", MakeScalarSeries(3), DimensionSpec::Uniform(3)});
            info.dependent_variables.push_back(
                DependentVariableInfo{"z", MakeScalarSeries(6)});
            return info;
        }

        inline BlockCreateInfo MakeValueRichCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-values";
            info.independent_variables.push_back(
                IndependentVariableInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"y", MakeScalarSeriesFrom({1.0, 2.0, 3.0}), DimensionSpec::Uniform(3)});
            info.dependent_variables.push_back(
                DependentVariableInfo{"z", MakeScalarSeriesFrom({100.0, 101.0, 102.0, 103.0, 104.0, 105.0})});
            return info;
        }

        // Three independents (2 × 3 × 4 uniform), two dependents
        inline BlockCreateInfo MakeThreeDimMultiDepCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-3d-multidep";
            info.independent_variables.push_back(
                IndependentVariableInfo{"a", MakeScalarSeriesFrom({1.0, 2.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"b", MakeScalarSeriesFrom({10.0, 20.0, 30.0}), DimensionSpec::Uniform(3)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"c", MakeScalarSeriesFrom({100.0, 200.0, 300.0, 400.0}), DimensionSpec::Uniform(4)});
            info.dependent_variables.push_back(
                DependentVariableInfo{"p", MakeScalarSeries(2 * 3 * 4)});
            info.dependent_variables.push_back(
                DependentVariableInfo{"q", MakeScalarSeries(2 * 3 * 4)});
            return info;
        }

        // Single independent (edge case)
        inline BlockCreateInfo MakeSingleIndependentCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-single";
            info.independent_variables.push_back(
                IndependentVariableInfo{"x", MakeScalarSeriesFrom({10.0, 20.0, 30.0}), DimensionSpec::Uniform(3)});
            info.dependent_variables.push_back(
                DependentVariableInfo{"z", MakeScalarSeriesFrom({100.0, 200.0, 300.0})});
            return info;
        }

        // =====================================================================
        // Fixtures: string-typed  (2 × 2)
        // =====================================================================

        inline BlockCreateInfo MakeStringTypedCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-strings";

            CellSeries xs = CellSeries::Scalars<std::string>(2);
            xs.scalar_at<std::string>(0) = "alpha";
            xs.scalar_at<std::string>(1) = "beta";
            info.independent_variables.push_back(
                IndependentVariableInfo{"sx", std::move(xs), DimensionSpec::Uniform(2)});

            CellSeries ys = CellSeries::Scalars<std::string>(2);
            ys.scalar_at<std::string>(0) = "one";
            ys.scalar_at<std::string>(1) = "two";
            info.independent_variables.push_back(
                IndependentVariableInfo{"sy", std::move(ys), DimensionSpec::Uniform(2)});

            info.dependent_variables.push_back(
                DependentVariableInfo{"sz",
                    MakeStringScalarSeriesFrom({"A", "B", "C", "D"})});
            return info;
        }

        // =====================================================================
        // Fixtures: vector / matrix cell kinds  (2 × 2)
        // =====================================================================

        inline BlockCreateInfo MakeVectorCellCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-vectors";
            info.independent_variables.push_back(
                IndependentVariableInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"y", MakeScalarSeriesFrom({1.0, 2.0}), DimensionSpec::Uniform(2)});
            info.dependent_variables.push_back(
                DependentVariableInfo{"vecs", MakeVectorSeries(4, 3)});
            return info;
        }

        inline BlockCreateInfo MakeMatrixCellCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-matrices";
            info.independent_variables.push_back(
                IndependentVariableInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"y", MakeScalarSeriesFrom({1.0, 2.0}), DimensionSpec::Uniform(2)});
            info.dependent_variables.push_back(
                DependentVariableInfo{"mats", MakeMatrixSeries(4, 2, 2)});
            return info;
        }

        // =====================================================================
        // Fixtures: jagged
        // =====================================================================

        inline BlockCreateInfo MakeJaggedCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-jagged";
            info.independent_variables.push_back(
                IndependentVariableInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"y", MakeScalarSeriesFrom({1.0, 2.0, 3.0}), DimensionSpec::Jagged({1, 2})});
            info.dependent_variables.push_back(
                DependentVariableInfo{"z", MakeScalarSeriesFrom({100.0, 101.0, 102.0})});
            return info;
        }

        inline BlockCreateInfo MakeInterleavedCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-interleaved";
            info.independent_variables.push_back(
                IndependentVariableInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"y", MakeScalarSeriesFrom({1.0, 2.0, 3.0}), DimensionSpec::Jagged({1, 2})});
            info.independent_variables.push_back(
                IndependentVariableInfo{"z", MakeScalarSeriesFrom({100.0, 200.0}), DimensionSpec::Uniform(2)});
            info.dependent_variables.push_back(
                DependentVariableInfo{"w", MakeScalarSeriesFrom({1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0})});
            return info;
        }

        // Jagged + vector dependent
        inline BlockCreateInfo MakeJaggedVectorCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-jagged-vectors";
            info.independent_variables.push_back(
                IndependentVariableInfo{"x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(
                IndependentVariableInfo{"y", MakeScalarSeriesFrom({1.0, 2.0, 3.0}), DimensionSpec::Jagged({1, 2})});
            info.dependent_variables.push_back(
                DependentVariableInfo{"v", MakeVectorSeries(3, 2)});
            return info;
        }
    } // namespace block_fixtures
} // namespace xdataset

#endif // BLOCK_FIXTURES_H
