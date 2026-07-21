#ifndef XDT_UNIT_H
#define XDT_UNIT_H

#include <llnl-units/units.hpp>
#include <string>

namespace xdataset
{

// =========================================================================
//  Unit — a physical unit type that only accepts REL-defined strings
// =========================================================================
//
//  Wraps llnl-units internally but restricts input parsing to the REL unit
//  vocabulary.  Construction by users goes through `Unit::parse()` which
//  validates against the REL lookup tables; construction from canonical base
//  units (used internally by operator*/operator/ /
//  canonicalize) bypasses validation — those are always valid SI.
// =========================================================================

/// Result of best_display: how to convert a raw value for display.
struct UnitScale {
    double scale;       ///< Multiply raw value by this to get display value.
    std::string name;   ///< Display-unit string (empty if dimensionless).
};

class Unit
{
public:
    // ---- construction ---------------------------------------------------

    /// Default: dimensionless, multiplier 1.
    Unit();

    /// True when the unit is in canonical form (multiplier == 1, non-affine).
    bool is_canonical() const;

    /// Return a canonicalised copy (multiplier absorbed, unit = base_units).
    Unit canonicalized() const;

    // ---- queries --------------------------------------------------------

    /// Physical multiplier (value scale factor).
    double multiplier() const;

    /// True when base_units has the e_flag set (degC, degF, etc.).
    bool is_affine() const;

    /// True when a and b represent the same physical dimension.
    bool same_dimension(const Unit& other) const;

    /// True when the physical dimension is empty (regardless of multiplier).
    bool has_dimension() const;

    /// True when dimensionless AND multiplier == 1.
    bool is_dimensionless() const;

    // ---- string conversion ----------------------------------------------

    /// Human-readable string.  Tries REL vocabulary first, falls back to
    /// llnl-units' native to_string.
    std::string to_string() const;

    // ---- display -------------------------------------------------------

    /// Given a raw value in this unit, pick the best display scale.
    /// Returns {scale, display_unit_string}.  Display value = raw * scale.
    /// For example, 1e9 Hz → {1e-9, "GHz"}, 0.002 V → {1000, "mV"}.
    UnitScale best_display(double value) const;

    // ---- static factory ------------------------------------------------

    /// Parse a REL unit string.  Throws std::invalid_argument when the
    /// string is not in the REL vocabulary.
    static Unit parse(const std::string& s);

    /// Access the underlying llnl precise_unit (e.g. for units::convert).
    const units::precise_unit& raw() const { return unit_; }

    // ---- arithmetic on dimensions (inputs must be canonical) ------------

    Unit operator*(const Unit& other) const;
    Unit operator/(const Unit& other) const;

    // ---- comparison ----------------------------------------------------

    bool operator==(const Unit& other) const;
    bool operator!=(const Unit& other) const;

private:
    // Private constructor from llnl precise_unit — only for internal use
    // by canonicalize / multiply_dim / etc.
    explicit Unit(units::precise_unit pu);

    units::precise_unit unit_;
};



} // namespace xdataset

#endif // XDT_UNIT_H
