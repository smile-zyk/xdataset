#include "block.h"
#include "unit.h"

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
    DataSeries doubles(const std::vector<double>& values, const Unit& u = Unit())
    {
        return DataSeries::CreateScalarFromVector<double>(values, u);
    }

    DataSeries vectors(std::size_t rows, Index width, const Unit& u = Unit())
    {
        DataSeries s(DataKind::kVector, DTypeTag::kReal, {width});
        s.set_unit(u);
        s.resize(rows);
        for (std::size_t i = 0; i < rows; ++i)
            for (Index j = 0; j < width; ++j)
                s.vector_at<double>(static_cast<Index>(i))(j) = static_cast<double>(static_cast<Index>(i) * width + j + 1);
        return s;
    }

    DataSeries matrices(std::size_t rows, Index r, Index c, const Unit& u = Unit())
    {
        DataSeries s(DataKind::kMatrix, DTypeTag::kReal, {r, c});
        s.set_unit(u);
        s.resize(rows);
        for (std::size_t i = 0; i < rows; ++i)
            for (Index ri = 0; ri < r; ++ri)
                for (Index ci = 0; ci < c; ++ci)
                    s.matrix_at<double>(static_cast<Index>(i))(ri, ci) = static_cast<double>(static_cast<Index>(i) * r * c + ri * c + ci + 1);
        return s;
    }

    DataSeries strings(const std::vector<std::string>& vals, const Unit& u = Unit())
    {
        DataSeries s = DataSeries::CreateScalar<std::string>(vals.size(), u);
        for (std::size_t i = 0; i < vals.size(); ++i)
            s.scalar_at<std::string>(static_cast<Index>(i)) = vals[i];
        return s;
    }

    void print_row(const DataFrameRow& r, const std::string& label = "")
    {
        std::cout << "  " << label << r.FormatMultiIndex() << " |";
        for (const auto& f : r.fields)
            std::cout << " " << f.to_string();
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
            auto m = Unit::parse("meter");
            auto s = Unit::parse("sec");
            auto V = Unit::parse("V");

            BlockCreateInfo info;
            info.independent_specs.push_back({"a", doubles({1.0, 2.0}, m), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"b", doubles({10.0, 20.0, 30.0}, s), DimensionSpec::Regular(3)});
            info.dependent_specs.push_back({"c", doubles({100.0, 101.0, 102.0, 103.0, 104.0, 105.0}, V)});
            Block block(info);

            const DataFrame& t = block.GetOrCreateDataFrame();
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
            info.independent_specs.push_back({"city", strings({"Paris", "London"}), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"unit", strings({"kg", "L"}), DimensionSpec::Regular(2)});
            info.dependent_specs.push_back({"val", strings({"A", "B", "C", "D"})});
            Block block(info);

            const DataFrame& t = block.GetOrCreateDataFrame();
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
            info.independent_specs.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"y", doubles({1.0, 2.0}), DimensionSpec::Regular(2)});
            info.dependent_specs.push_back({"vec", vectors(4, 3, Unit::parse("W"))});
            Block block(info);

            const DataFrame& t = block.GetOrCreateDataFrame();
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
            info.independent_specs.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"y", doubles({1.0, 2.0}), DimensionSpec::Regular(2)});
            info.dependent_specs.push_back({"mat", matrices(4, 2, 2, Unit::parse("meter"))});
            Block block(info);

            const DataFrame& t = block.GetOrCreateDataFrame();
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
            info.independent_specs.push_back({"a", doubles({1.0, 2.0}), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"b", doubles({10.0, 20.0, 30.0}), DimensionSpec::Regular(3)});
            info.independent_specs.push_back({"c", doubles({100.0, 200.0, 300.0, 400.0}), DimensionSpec::Regular(4)});
            info.dependent_specs.push_back({"p", doubles(std::vector<double>(24, 0.0))});
            info.dependent_specs.push_back({"q", doubles(std::vector<double>(24, 0.0))});
            Block block(info);

            const DataFrame& t = block.GetOrCreateDataFrame();
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
            info.independent_specs.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"y", doubles({1.0, 2.0, 3.0}), DimensionSpec::Ragged({1, 2})});
            info.independent_specs.push_back({"z", doubles({100.0, 200.0}), DimensionSpec::Regular(2)});
            info.dependent_specs.push_back({"w", doubles({1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0})});
            info.dependent_specs.push_back({"v", vectors(6, 2, Unit::parse("Hz"))});
            Block block(info);

            const DataFrame& t = block.GetOrCreateDataFrame();
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
            info.independent_specs.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"y", doubles({1.0, 2.0, 3.0}), DimensionSpec::Regular(3)});
            info.dependent_specs.push_back({"z", doubles({100.0, 101.0, 102.0, 103.0, 104.0, 105.0})});
            Block block(info);

            const DataFrame& t = block.GetOrCreateDataFrame();
            std::cout << "total row_count(): " << t.row_count() << " (but only 2 are loaded below)" << std::endl;
            print_row(t.GetRow(0));
            print_row(t.GetRow(1));
            // rows 2-5 are never accessed -- their chunks are never loaded
            std::cout << "(rows 2-5 were never materialised)" << std::endl;
        }

        // =====================================================================
        // 8. DataArray.indep() chain
        // =====================================================================
        section("8. DataArray.indep() chain");
        {
            BlockCreateInfo info;
            info.independent_specs.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"y", doubles({1.0, 2.0, 3.0}), DimensionSpec::Ragged({1, 2})});
            info.independent_specs.push_back({"z", doubles({100.0, 200.0}), DimensionSpec::Regular(2)});
            info.dependent_specs.push_back({"w", doubles({1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0})});
            Block block(info);

            auto w = block.GetOrCreateDataArray("w");           // dependent
            std::cout << "w.kind() = " << (w.kind() == DataArrayKind::kDependent ? "dependent" : "independent") << std::endl;

            auto z_var = w.indep(1);                         // dependent w -> indep(1) = z
            std::cout << "w.indep(1).name() = " << z_var.name() << ", rank = " << z_var.multi_dimension_spec().rank() << std::endl;

            auto y_var = z_var.indep(2);                     // z -> indep(2) = y
            std::cout << "z.indep(2).name() = " << y_var.name() << ", rank = " << y_var.multi_dimension_spec().rank() << std::endl;

            auto by_name = w.indep("y");                     // by name
            std::cout << "w.indep(\"y\").name() = " << by_name.name() << std::endl;

            // grid model from indep(1)
            std::cout << "\nGrid from w.indep(1):" << std::endl;
            const DataFrame& zt = z_var.GetOrCreateDataFrame();
            std::cout << "  headers: ";
            for (const auto& h : zt.headers()) std::cout << h << " ";
            std::cout << "\n  rows: " << zt.row_count() << std::endl;
            print_row(zt.GetRow(0));
            print_row(zt.GetRow(5));
        }

        // =====================================================================
        // 9. DataArray.select() demo
        // =====================================================================
        section("9. DataArray.select()");
        {
            BlockCreateInfo info;
            info.independent_specs.push_back({"x", doubles({10.0, 20.0}), DimensionSpec::Regular(2)});
            info.independent_specs.push_back({"y", doubles({1.0, 2.0, 3.0}), DimensionSpec::Ragged({1, 2})});
            info.independent_specs.push_back({"z", doubles({100.0, 200.0}), DimensionSpec::Regular(2)});
            info.dependent_specs.push_back({"w", doubles({1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0})});
            Block block(info);

            auto w = block.GetOrCreateDataArray("w");

            // select where x=1, y=any, z=any -> collapses first dim
            auto sel = w.select({MultiIndexSelector::Equal(1),
                                  MultiIndexSelector::Any(),
                                  MultiIndexSelector::Any()});
            std::cout << "w.select(Equal(1), Any, Any) -> rank "
                      << sel.multi_dimension_spec().rank() << ", "
                      << sel.data().size() << " rows" << std::endl;
            const DataFrame& st = sel.GetOrCreateDataFrame();
            for (std::size_t i = 0; i < st.row_count(); ++i)
                print_row(st.GetRow(static_cast<Index>(i)));
        }

        // =====================================================================
        // 10. Measurement type inspection
        // =====================================================================
        section("10. Measurement type inspection");
        {
            Measurement d(3.14, Unit::parse("Ohm"));
            Measurement i(42, Unit::parse("F"));
            Measurement c(std::complex<double>(1.0, -2.0), Unit::parse("V"));
            Measurement s(std::string("hello"));

            std::cout << "Measurement(3.14 m/s):     "
                      << "dtype=" << d.dtype_name()
                      << "  ToString=\"" << d.to_string() << "\"" << std::endl;
            std::cout << "Measurement(42 kg):        "
                      << "dtype=" << i.dtype_name()
                      << "  ToString=\"" << i.to_string() << "\"" << std::endl;
            std::cout << "Measurement(1-2i V):       "
                      << "dtype=" << c.dtype_name()
                      << "  ToString=\"" << c.to_string() << "\"" << std::endl;
            std::cout << "Measurement(\"hello\"):   "
                      << "dtype=" << s.dtype_name()
                      << "  ToString=\"" << s.to_string() << "\"" << std::endl;

            std::cout << "Measurement(3.14).as_scalar<double>() = " << d.as_scalar<double>() << std::endl;
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
