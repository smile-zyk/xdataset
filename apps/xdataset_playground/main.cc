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

        BlockCreateInfo MakeInterleavedCreateInfo()
        {
            BlockCreateInfo info;
            info.name = "demo-interleaved";
            info.independent_variables.push_back(IndependentVariableCreateInfo{
                "x", MakeScalarSeriesFrom({10.0, 20.0}), DimensionSpec::Uniform(2)});
            info.independent_variables.push_back(IndependentVariableCreateInfo{
                "y", MakeScalarSeriesFrom({1.0, 2.0, 3.0}), DimensionSpec::Uniform(3)});
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

            return info;
        }
    } // namespace
} // namespace xdataset

int main()
{
    try
    {
        xdataset::Block block(xdataset::MakeInterleavedCreateInfo());

        const xdataset::TableData block_table = block.ToTableData();
        block_table.WriteToCsv("block_all_variables.csv");
        std::cout << "Wrote block table to block_all_variables.csv" << std::endl;

        for (const std::string& name : block.independents())
        {
            const std::shared_ptr<xdataset::VariableData> var_data =
                block.GetOrCreateVariableData(name);
            if (!var_data)
            {
                throw std::runtime_error("failed to create VariableData for independent: " + name);
            }
            const xdataset::TableData table = var_data->ToTableData();
            const std::string file_name = "variable_" + name + ".csv";
            table.WriteToCsv(file_name);
            std::cout << "Wrote independent variable table to " << file_name << std::endl;
        }

        for (const std::string& name : block.dependents())
        {
            const std::shared_ptr<xdataset::VariableData> var_data =
                block.GetOrCreateVariableData(name);
            if (!var_data)
            {
                throw std::runtime_error("failed to create VariableData for dependent: " + name);
            }
            const xdataset::TableData table = var_data->ToTableData();
            const std::string file_name = "variable_" + name + ".csv";
            table.WriteToCsv(file_name);
            std::cout << "Wrote dependent variable table to " << file_name << std::endl;
        }

        std::cout << "Done." << std::endl;
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
