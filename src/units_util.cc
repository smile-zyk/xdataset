#include "units_util.h"

#include <stdexcept>

namespace xdataset
{

    Unit parse_unit(const std::string& s)
    {
        Unit u = units::unit_from_string(s);
        if (units::isnan(u))
        {
            throw std::invalid_argument("Unrecognised unit string: '" + s + "'");
        }
        return u;
    }

    Unit canonicalize(const Unit& u)
    {
        // Drop multiplier and affine flag → pure coherent SI (base units only).
        return Unit(u.base_units().clear_e_flag());
    }

    bool is_affine(const Unit& u)
    {
        return u.base_units().has_e_flag();
    }

    bool same_dimension(const Unit& a, const Unit& b)
    {
        // has_same_base ignores e_flag / i_flag, so e.g. degC ≡ K.
        return a.has_same_base(b);
    }

    double multiplier_of(const Unit& u)
    {
        return u.multiplier();
    }

    Unit multiply_dim(const Unit& a, const Unit& b)
    {
        // Both inputs are already canonical (multiplier == 1), so we only
        // combine the base unit exponents; the result stays canonical.
        return Unit(a.base_units() * b.base_units());
    }

    Unit divide_dim(const Unit& a, const Unit& b)
    {
        return Unit(a.base_units() / b.base_units());
    }

    Unit pow_dim(const Unit& a, int n)
    {
        return Unit(a.base_units().pow(n));
    }

    std::string unit_string(const Unit& u)
    {
        return units::to_string(u);
    }

} // namespace xdataset
