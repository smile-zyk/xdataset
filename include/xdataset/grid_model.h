#ifndef GRID_MODEL_H
#define GRID_MODEL_H

#include <boost/variant.hpp>

#include <complex>
#include <functional>
#include <string>
#include <vector>

#include "cell_series.h"

namespace xdataset
{
    class CellSeries;

    // Typed value in a single grid column: covers all supported CellSeries element types.
    using GridField = boost::variant<double, int, std::complex<double>, std::string>;

    // ---------------------------------------------------------------------------
    // Singleton visitor: converts a GridField to its CSV-safe string representation.
    // ---------------------------------------------------------------------------
    struct GridFieldStringifyVisitor : public boost::static_visitor<std::string>
    {
        static const GridFieldStringifyVisitor& Instance();

        std::string operator()(double v) const;
        std::string operator()(int v) const;
        std::string operator()(const std::complex<double>& v) const;
        std::string operator()(const std::string& v) const;
    };

    // ---------------------------------------------------------------------------
    // Singleton visitor: extracts typed information from a GridField.
    //
    // One-shot type queries (no Apply needed):
    //   GridFieldGetVisitor::TypeOf(field)   -> DTypeTag
    //   GridFieldGetVisitor::IsInt(field)    -> bool
    //
    // Chained value extraction:
    //   int x = GridFieldGetVisitor::Apply(field).as_int();
    //
    // Full access:
    //   auto& v = GridFieldGetVisitor::Apply(field);
    //   switch (v.type()) { case DTypeTag::kInteger: use(v.as_int()); ... }
    // ---------------------------------------------------------------------------
    class GridFieldGetVisitor : public boost::static_visitor<void>
    {
    public:
        static const GridFieldGetVisitor& Apply(const GridField& field);

        // --- one-shot type helpers (do not mutate singleton state) ---
        static DTypeTag TypeOf(const GridField& field);
        static bool IsDouble(const GridField& field);
        static bool IsInt(const GridField& field);
        static bool IsComplex(const GridField& field);
        static bool IsString(const GridField& field);

        // --- typed accessors (valid after Apply()) ---
        DTypeTag type() const { return type_; }

        const double&               as_double()  const { return d_val_; }
        const int&                  as_int()     const { return i_val_; }
        const std::complex<double>& as_complex() const { return c_val_; }
        const std::string&          as_string()  const { return s_val_; }

        void operator()(double v);
        void operator()(int v);
        void operator()(const std::complex<double>& v);
        void operator()(const std::string& v);

    private:
        GridFieldGetVisitor() = default;

        DTypeTag type_ = DTypeTag::kReal;
        double              d_val_ = 0.0;
        int                 i_val_ = 0;
        std::complex<double> c_val_{};
        std::string         s_val_;
    };

    // Free helper: stringify a single GridField.
    inline std::string GridFieldToString(const GridField& field)
    {
        return boost::apply_visitor(GridFieldStringifyVisitor::Instance(), field);
    }

    // =========================================================================
    // GridRow — a single row in a GridModel table
    // =========================================================================
    struct GridRow
    {
        std::vector<Index>  multi_index;
        std::vector<GridField>    fields;

        std::string FormatMultiIndex() const;
    };

    // =========================================================================
    // GridModel — lazy-loading tabular container
    // =========================================================================
    //
    // Rows are loaded on demand in fixed-size chunks.  Derived classes
    // (BlockGridModel, VariableGridModel) provide the RowGenerator that
    // populates rows from their respective data sources.
    // =========================================================================

    class GridModel
    {
    public:
        const std::vector<std::string>& headers()   const { return headers_;   }
        std::size_t                     row_count() const { return total_rows_; }

        const GridRow& GetRow(Index row) const;

        std::string ToCsv() const;
        void        WriteToCsv(const std::string& file_path) const;

    protected:
        using RowGenerator = std::function<std::vector<GridRow>(
            Index start_row,
            Index end_row)>;

        GridModel() = default;

        void Configure(std::vector<std::string> headers,
                       std::size_t total_rows,
                       RowGenerator generator,
                       std::size_t chunk_size = 256);

    private:
        void EnsureChunkLoaded(Index chunk_idx) const;

        std::vector<std::string>   headers_;
        std::size_t                total_rows_  = 0;
        std::size_t                chunk_size_  = 256;
        RowGenerator               generator_;

        mutable std::vector<GridRow>  rows_;
        mutable std::vector<bool>     loaded_chunks_;
    };

    // =========================================================================
    // BlockGridModel — GridModel backed by a Block's independent/dependent data
    // =========================================================================
    class Block;
    class BlockGridModel : public GridModel
    {
    public:
        explicit BlockGridModel(const Block& block);
    };

    // =========================================================================
    // VariableGridModel — GridModel backed by a Variable's data
    // =========================================================================
    class Variable;
    class VariableGridModel : public GridModel
    {
    public:
        explicit VariableGridModel(const Variable& variable);
    };

} // namespace xdataset

#endif  // GRID_MODEL_H
