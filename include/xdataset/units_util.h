#ifndef UNITS_UTIL_H
#define UNITS_UTIL_H

#include <llnl-units/units.hpp>
#include <string>

namespace xdataset
{
    using Unit = units::precise_unit;

    // Parse a unit string (e.g. "m/s", "Hz", "kg*m^-2").  Returns an error
    // unit (units::isnan() == true) when the string is not recognised.
    Unit parse_unit(const std::string& s);

    // Return the canonical SI form of a unit: multiplier forced to 1 and
    // e_flag cleared so that affine units (degC/degF) collapse to K.
    Unit canonicalize(const Unit& u);

    // True when base_units has the e_flag set (degC, degF, etc.).
    bool is_affine(const Unit& u);

    // True when u is in canonical form (multiplier == 1, non-affine).
    bool is_canonical(const Unit& u);

    // True when a and b represent the same physical dimension.
    // Uses has_same_base so that affine flags are ignored.
    bool same_dimension(const Unit& a, const Unit& b);

    // True when u is dimensionless (no physical dimension).
    bool is_dimensionless(const Unit& u);

    // Convenience: multiplier, but separately from the canonical-SI logic.
    double multiplier_of(const Unit& u);

    // Derive the canonical unit for a * b, a / b, and a^n.
    // The inputs are expected to already be canonical (multiplier == 1).
    Unit multiply_dim(const Unit& a, const Unit& b);
    Unit divide_dim(const Unit& a, const Unit& b);
    Unit pow_dim(const Unit& a, int n);

    // Human-readable unit name via llnl-units' reverse-lookup table.
    // Recovers coherent SI derived-unit names (Hz, Pa, N, ...) for free.
    std::string unit_string(const Unit& u);

} // namespace xdataset

#endif // UNITS_UTIL_H
