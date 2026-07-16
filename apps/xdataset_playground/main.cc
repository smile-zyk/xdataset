#include "block.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace xdataset;

// =============================================================================
// Helpers
// =============================================================================

namespace
{
    CellSeries doubles(const std::vector<double>& values)
    {
        return CellSeries::ScalarsFrom<double>(values);
    }

    CellSeries vectors(std::size_t rows, Index width)
    {
        CellSeries s = CellSeries::Vectors<double>(rows, width);
        for (std::size_t i = 0; i < rows; ++i)
            for (Index j = 0; j < width; ++j)
                s.vector_at<double>(static_cast<Index>(i))(j) = static_cast<double>(static_cast<Index>(i) * width + j + 1);
        return s;
    }

    CellSeries matrices(std::size_t rows, Index r, Index c)
    {
        CellSeries s = CellSeries::Matrices<double>(rows, r, c);
        for (std::size_t i = 0; i < rows; ++i)
            for (Index ri = 0; ri < r; ++ri)
                for (Index ci = 0; ci < c; ++ci)
                    s.matrix_at<double>(static_cast<Index>(i))(ri, ci) = static_cast<double>(static_cast<Index>(i) * r * c + ri * c + ci + 1);
        return s;
    }

    CellSeries strings(const std::vector<std::string>& vals)
    {
        CellSeries s = CellSeries::Scalars<std::string>(vals.size());
        for (std::size_t i = 0; i < vals.size(); ++i)
            s.scalar_at<std::string>(static_cast<Index>(i)) = vals[i];
        return s;
    }

    void print_row(const GridRow& r, const std::string& label = "")
    {
        std::cout << "  " << label << r.FormatMultiIndex() << " |";
        for (const auto& f : r.fields)
            std::cout << " " << GridFieldToString(f);
        std::cout << std::endl;
    }

    void section(const std::string& title)
    {
        std::cout << "\n=== " << title << " ===" << std::endl;
    }
} // namespace

// =============================================================================
int main()
{
    try
    {
        // =====================================================================
        // 1. Simple regular block (2 x 3)
        // =====================================================================
        section("1. Simple 2x3 regular block");
        {
            BlockCreateInfo info;
            info.name = "simple";
            info.independent_variables.push_back({"a", doubles({1.0, 2.0}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"b", doubles({10.0, 20.0, 30.0}), DimensionSpec::Regular(3)});
            info.dependent_variables.push_back({"c", doubles({100.0, 101.0, 102.0, 103.0, 104.0, 105.0})});
            Block block(info);

            const GridModel& t = block.grid_model();
            std::cout << "headers: ";
            for (const auto& h : t.headers()) std::cout << h << " ";
            std::cout << "\nrows: " << t.row_count() << std::endl;
            // print first & last row
            print_row(t.GetRow(0), "first: ");
            print_row(t.GetRow(5), "last:  ");
            t.WriteToCsv("output/demo_simple.csv");
            std::cout << "-> written to demo_simple.csv" << std::endl;
        }

        // =====================================================================
        // 2. String-typed block
        // =====================================================================
        section("2. String-typed block");
        {
            BlockCreateInfo info;
            info.name = "strings";
            info.independent_variables.push_back({"city", strings({"Paris", "London"}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"unit", strings({"kg", "L"}), DimensionSpec::Regular(2)});
            info.dependent_variables.push_back({"val", strings({"A", "B", "C", "D"})});
            Block block(info);

            const GridModel& t = block.grid_model();
            print_row(t.GetRow(0));
            print_row(t.GetRow(1));
            print_row(t.GetRow(2));
            print_row(t.GetRow(3));
            t.WriteToCsv("output/demo_strings.csv");
            std::cout << "-> written to demo_strings.csv" << std::endl;
        }

        // =====================================================================
        // 3. Vector & matrix cell blocks
        // =====================================================================
        section("3. Vector cell block (2x2 -> 4 rows, vec width=3)");
        {
            BlockCreateInfo info;
            info.name = "vectors";
            info.independent_variables.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"y", doubles({1.0, 2.0}), DimensionSpec::Regular(2)});
            info.dependent_variables.push_back({"vec", vectors(4, 3)});
            Block block(info);

            const GridModel& t = block.grid_model();
            for (const auto& h : t.headers()) std::cout << "  " << h;
            std::cout << std::endl;
            for (std::size_t i = 0; i < t.row_count(); ++i)
                print_row(t.GetRow(static_cast<Index>(i)));
            t.WriteToCsv("output/demo_vectors.csv");
            std::cout << "-> written to demo_vectors.csv" << std::endl;
        }

        section("4. Matrix cell block (2x2 -> 4 rows, mat 2x2)");
        {
            BlockCreateInfo info;
            info.name = "matrices";
            info.independent_variables.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"y", doubles({1.0, 2.0}), DimensionSpec::Regular(2)});
            info.dependent_variables.push_back({"mat", matrices(4, 2, 2)});
            Block block(info);

            const GridModel& t = block.grid_model();
            for (const auto& h : t.headers()) std::cout << "  " << h;
            std::cout << std::endl;
            for (std::size_t i = 0; i < t.row_count(); ++i)
                print_row(t.GetRow(static_cast<Index>(i)));
            t.WriteToCsv("output/demo_matrices.csv");
            std::cout << "-> written to demo_matrices.csv" << std::endl;
        }

        // =====================================================================
        // 5. Three-dimensional block with two dependents
        // =====================================================================
        section("5. 3D block (2x3x4) with two dependents");
        {
            BlockCreateInfo info;
            info.name = "3d";
            info.independent_variables.push_back({"a", doubles({1.0, 2.0}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"b", doubles({10.0, 20.0, 30.0}), DimensionSpec::Regular(3)});
            info.independent_variables.push_back({"c", doubles({100.0, 200.0, 300.0, 400.0}), DimensionSpec::Regular(4)});
            info.dependent_variables.push_back({"p", doubles(std::vector<double>(24, 0.0))});
            info.dependent_variables.push_back({"q", doubles(std::vector<double>(24, 0.0))});
            Block block(info);

            const GridModel& t = block.grid_model();
            std::cout << "total rows: " << t.row_count() << ", columns: " << t.headers().size() << std::endl;
            print_row(t.GetRow(0),  "first:  ");
            print_row(t.GetRow(23), "last:   ");
            t.WriteToCsv("output/demo_3d.csv");
            std::cout << "-> written to demo_3d.csv" << std::endl;
        }

        // =====================================================================
        // 6. Ragged + interleaved (original demo)
        // =====================================================================
        section("6. Ragged-interleaved block (x x y(Ragged) x z)");
        {
            BlockCreateInfo info;
            info.name = "interleaved";
            info.independent_variables.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"y", doubles({1.0, 2.0, 3.0}), DimensionSpec::Ragged({1, 2})});
            info.independent_variables.push_back({"z", doubles({100.0, 200.0}), DimensionSpec::Regular(2)});
            info.dependent_variables.push_back({"w", doubles({1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0})});
            info.dependent_variables.push_back({"v", vectors(6, 2)});
            Block block(info);

            const GridModel& t = block.grid_model();
            for (const auto& h : t.headers()) std::cout << "  " << h;
            std::cout << std::endl;
            for (std::size_t i = 0; i < t.row_count(); ++i)
                print_row(t.GetRow(static_cast<Index>(i)));
            t.WriteToCsv("output/demo_jagged.csv");
            std::cout << "-> written to demo_jagged.csv" << std::endl;
        }

        // =====================================================================
        // 7. Lazy loading demo -- only access a few rows
        // =====================================================================
        section("7. Lazy loading -- only first 2 rows accessed from 2x3 grid");
        {
            BlockCreateInfo info;
            info.name = "lazy";
            info.independent_variables.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"y", doubles({1.0, 2.0, 3.0}), DimensionSpec::Regular(3)});
            info.dependent_variables.push_back({"z", doubles({100.0, 101.0, 102.0, 103.0, 104.0, 105.0})});
            Block block(info);

            const GridModel& t = block.grid_model();
            std::cout << "total row_count(): " << t.row_count() << " (but only 2 are loaded below)" << std::endl;
            print_row(t.GetRow(0));
            print_row(t.GetRow(1));
            // rows 2-5 are never accessed -- their chunks are never loaded
            std::cout << "(rows 2-5 were never materialised)" << std::endl;
        }

        // =====================================================================
        // 8. Variable.indep() chain
        // =====================================================================
        section("8. Variable.indep() chain");
        {
            BlockCreateInfo info;
            info.name = "indep";
            info.independent_variables.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"y", doubles({1.0, 2.0, 3.0}), DimensionSpec::Ragged({1, 2})});
            info.independent_variables.push_back({"z", doubles({100.0, 200.0}), DimensionSpec::Regular(2)});
            info.dependent_variables.push_back({"w", doubles({1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0})});
            Block block(info);

            auto w = block.GetOrCreateVariable("w");           // dependent
            std::cout << "w.kind() = " << (w->kind() == VariableKind::kDependent ? "dependent" : "independent") << std::endl;

            auto z_var = w->indep(1);                         // dependent w -> indep(1) = z
            std::cout << "w.indep(1).name() = " << z_var->name() << ", rank = " << z_var->multi_dimension_spec().rank() << std::endl;

            auto y_var = z_var->indep(2);                     // z -> indep(2) = y
            std::cout << "z.indep(2).name() = " << y_var->name() << ", rank = " << y_var->multi_dimension_spec().rank() << std::endl;

            auto by_name = w->indep("y");                     // by name
            std::cout << "w.indep(\"y\").name() = " << by_name->name() << std::endl;

            // grid model from indep(1)
            std::cout << "\nGrid from w.indep(1):" << std::endl;
            const GridModel& zt = z_var->grid_model();
            std::cout << "  headers: ";
            for (const auto& h : zt.headers()) std::cout << h << " ";
            std::cout << "\n  rows: " << zt.row_count() << std::endl;
            print_row(zt.GetRow(0));
            print_row(zt.GetRow(5));
        }

        // =====================================================================
        // 9. Variable.select() demo
        // =====================================================================
        section("9. Variable.select()");
        {
            BlockCreateInfo info;
            info.name = "select";
            info.independent_variables.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_variables.push_back({"y", doubles({1.0, 2.0, 3.0}), DimensionSpec::Ragged({1, 2})});
            info.independent_variables.push_back({"z", doubles({100.0, 200.0}), DimensionSpec::Regular(2)});
            info.dependent_variables.push_back({"w", doubles({1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0})});
            Block block(info);

            auto w = block.GetOrCreateVariable("w");

            // select where x=1, y=any, z=any -> collapses first dim
            auto sel = w->select({MultiIndexSelector::Equal(1),
                                  MultiIndexSelector::Any(),
                                  MultiIndexSelector::Any()});
            std::cout << "w.select(Equal(1), Any, Any) -> rank "
                      << sel->multi_dimension_spec().rank() << ", "
                      << sel->data().size() << " rows" << std::endl;
            const GridModel& st = sel->grid_model();
            for (std::size_t i = 0; i < st.row_count(); ++i)
                print_row(st.GetRow(static_cast<Index>(i)));
        }

        // =====================================================================
        // 10. GridField type inspection
        // =====================================================================
        section("10. GridField type inspection");
        {
            GridField d(3.14);
            GridField i(42);
            GridField c(std::complex<double>(1.0, -2.0));
            GridField s(std::string("hello"));

            std::cout << "GridField(3.14):        "
                      << "TypeOf=" << static_cast<int>(GridFieldGetVisitor::TypeOf(d))
                      << "  ToString=\"" << GridFieldToString(d) << "\"" << std::endl;
            std::cout << "GridField(42):          "
                      << "IsInt? " << (GridFieldGetVisitor::IsInt(i) ? "yes" : "no")
                      << "  ToString=\"" << GridFieldToString(i) << "\"" << std::endl;
            std::cout << "GridField(1-2i):        "
                      << "IsComplex? " << (GridFieldGetVisitor::IsComplex(c) ? "yes" : "no")
                      << "  ToString=\"" << GridFieldToString(c) << "\"" << std::endl;
            std::cout << "GridField(\"hello\"):   "
                      << "IsString? " << (GridFieldGetVisitor::IsString(s) ? "yes" : "no")
                      << "  ToString=\"" << GridFieldToString(s) << "\"" << std::endl;

            auto& v = GridFieldGetVisitor::Apply(d);
            std::cout << "GridFieldGetVisitor::Apply(3.14).as_double() = " << v.as_double() << std::endl;
        }

        std::cout << "\nDone." << std::endl;
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
