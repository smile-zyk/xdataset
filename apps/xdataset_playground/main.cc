#include "block.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xdataset
{
    namespace
    {
        CellSeries MakeScalarSeriesFrom(const std::vector<double>& values)
        {
            return CellSeries::ScalarsFrom<double>(values);
        }

        CellSeries MakeListSeriesFromRows(std::size_t rows)
        {
            CellSeries lists = CellSeries::Vectors<double>(rows, 3);
            for (std::size_t i = 0; i < rows; ++i)
            {
                const double base = static_cast<double>(i) * 10.0;
                lists.vector_at<double>(i) << base + 1.0, base + 2.0, base + 3.0;
            }
            return lists;
        }

        CellSeries MakeMatrixSeriesFromRows(std::size_t rows)
        {
            CellSeries mats = CellSeries::Matrices<double>(rows, 2, 2);
            for (std::size_t i = 0; i < rows; ++i)
            {
                const double base = static_cast<double>(i) * 100.0;
                mats.matrix_at<double>(i) << base + 1.0, base + 2.0,
                                             base + 3.0, base + 4.0;
            }
            return mats;
        }

        BlockCreateInfo MakeInterleavedCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-interleaved";
            info.independent_variables.push_back(IndependentVariableCreateInfo{
                "x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(IndependentVariableCreateInfo{
                "y",
                MakeScalarSeriesFrom({1.3, 2.8, 4.9, 7.2, 11.6, 16.4}),
                DimensionSpec::Jagged({2, 4})});
            info.independent_variables.push_back(IndependentVariableCreateInfo{
                "z", MakeScalarSeriesFrom({100.0, 200.0}), DimensionSpec::Uniform(2)});
            info.dependent_variables.push_back(
                DependentVariableCreateInfo{"w",
                                            MakeScalarSeriesFrom({1000.0,
                                                                  1001.0,
                                                                  1002.0,
                                                                  1003.0,
                                                                  1004.0,
                                                                  1005.0,
                                                                  1006.0,
                                                                  1007.0,
                                                                  1008.0,
                                                                  1009.0,
                                                                  1010.0,
                                                                  1011.0})});
            info.dependent_variables.push_back(
                DependentVariableCreateInfo{"v",
                                            MakeScalarSeriesFrom({2000.0,
                                                                  2001.0,
                                                                  2002.0,
                                                                  2003.0,
                                                                  2004.0,
                                                                  2005.0,
                                                                  2006.0,
                                                                  2007.0,
                                                                  2008.0,
                                                                  2009.0,
                                                                  2010.0,
                                                                  2011.0})});
            info.dependent_variables.push_back(
                DependentVariableCreateInfo{"list", MakeListSeriesFromRows(12)});
            info.dependent_variables.push_back(
                DependentVariableCreateInfo{"matrix", MakeMatrixSeriesFromRows(12)});

            return info;
        }
    } // namespace
} // namespace xdataset

int main()
{
    try
    {
        xdataset::Block block(xdataset::MakeInterleavedCreateInfo());

        const xdataset::TableData& block_table = block.GetOrCreateTableData();
        block_table.WriteToCsv("block_all_variables.csv");
        std::cout << "Wrote block table to block_all_variables.csv" << std::endl;

        for (const std::string& name : block.independents())
        {
            const std::shared_ptr<xdataset::Variable> variable =
                block.GetOrCreateVariable(name);
            if (!variable)
            {
                throw std::runtime_error("failed to create Variable for independent: " + name);
            }
            const xdataset::TableData& table = variable->GetOrCreateTableData();
            const std::string file_name = "variable_" + name + ".csv";
            table.WriteToCsv(file_name);
            std::cout << "Wrote independent variable table to " << file_name << std::endl;
        }

        for (const std::string& name : block.dependents())
        {
            const std::shared_ptr<xdataset::Variable> variable =
                block.GetOrCreateVariable(name);
            if (!variable)
            {
                throw std::runtime_error("failed to create Variable for dependent: " + name);
            }
            const xdataset::TableData& table = variable->GetOrCreateTableData();
            const std::string file_name = "variable_" + name + ".csv";
            table.WriteToCsv(file_name);
            std::cout << "Wrote dependent variable table to " << file_name << std::endl;
        }

        auto variable = block.GetOrCreateVariable("w");

        auto indep_1 = variable->indep(1);
        //output csv of indep_1
        const auto& indep_1_table = indep_1->GetOrCreateTableData();
        indep_1_table.WriteToCsv("variable_w_indep_1.csv");
        auto indep_2 = variable->indep(2);
        //output csv of indep_2
        const auto& indep_2_table = indep_2->GetOrCreateTableData();
        indep_2_table.WriteToCsv("variable_w_indep_2.csv");

        auto indep_variable = variable->indep("z");
        auto indep_3 = indep_variable->indep(1);
        //output csv of indep_3
        const auto& indep_3_table = indep_3->GetOrCreateTableData();
        indep_3_table.WriteToCsv("variable_w_indep_z_indep_1.csv");

        std::cout << "Done." << std::endl;
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
