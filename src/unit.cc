#include "unit.h"

#include <map>
#include <sstream>
#include <stdexcept>


namespace xdataset
{

    // =========================================================================
    //  REL vocabulary — only these strings are accepted by parse()
    // =========================================================================

    namespace
    {

        // --- scale factors (case-sensitive) --------------------------------------

        const std::map<std::string, double> kScaleFactors = {
            {"T", 1e12},
            {"G", 1e9},
            {"M", 1e6},
            {"K", 1e3},
            {"k", 1e3},
            {"_", 1.0},
            {"m", 1e-3},
            {"u", 1e-6},
            {"n", 1e-9},
            {"p", 1e-12},
            {"f", 1e-15},
            {"a", 1e-18},
        };

        // --- known physical units -> llnl-compatible string -----------------------

        const std::map<std::string, std::string> kUnitMap = {
            {"Hz", "Hz"},
            {"Ohm", "ohm"},
            {"Ohms", "ohm"},
            {"S", "S"},
            {"F", "F"},
            {"H", "H"},
            {"meter", "m"},
            {"meters", "m"},
            {"metre", "m"},
            {"metres", "m"},
            {"sec", "s"},
            {"V", "V"},
            {"A", "A"},
            {"W", "W"},
        };

        // --- predefined scaled units (exact match — no scale factor allowed) ------

        struct PredefUnit
        {
            double mult;
            std::string llnl;
        };

        const std::map<std::string, PredefUnit> kPredefUnits = 
        {
            {"mil", {2.54e-5, "m"}},
            {"mils", {2.54e-5, "m"}},
            {"in", {2.54e-2, "m"}},
            {"ft", {12 * 2.54e-2, "m"}},
            {"mi", {5280 * 12 * 2.54e-2, "m"}},
            {"cm", {1.0e-2, "m"}},
            {"PHz", {1.0e15, "Hz"}},
            {"dB", {1.0, ""}},
            {"nmi", {1852, "m"}},
        };

        // =========================================================================
        //  Token conversion: REL syntax -> llnl-compatible string
        // =========================================================================

        bool try_scale_unit(const std::string& token, double& mult, std::string& llnl_unit)
        {
            const std::size_t n = token.size();
            for (std::size_t len = n; len > 0; --len)
            {
                std::string prefix = token.substr(0, len);
                auto sit = kScaleFactors.find(prefix);
                if (sit == kScaleFactors.end())
                    continue;

                std::string suffix = token.substr(len);
                if (suffix.empty())
                {
                    mult = sit->second;
                    llnl_unit.clear();
                    return true;
                }
                auto uit = kUnitMap.find(suffix);
                if (uit != kUnitMap.end())
                {
                    mult = sit->second;
                    llnl_unit = uit->second;
                    return true;
                }
            }
            return false;
        }

        std::string convert_token(const std::string& token)
        {
            if (token.empty())
                return token;

            // 1) predefined scaled unit (exact match)
            auto pit = kPredefUnits.find(token);
            if (pit != kPredefUnits.end())
            {
                if (pit->second.llnl.empty())
                {
                    std::ostringstream oss;
                    oss << pit->second.mult;
                    return oss.str();
                }
                std::ostringstream oss;
                oss << "(" << pit->second.mult << "*" << pit->second.llnl << ")";
                return oss.str();
            }

            // 2) known unit without scale factor
            auto uit = kUnitMap.find(token);
            if (uit != kUnitMap.end())
                return uit->second;

            // 3) scale factor + known unit
            double mult = 1.0;
            std::string llnl_unit;
            if (try_scale_unit(token, mult, llnl_unit))
            {
                if (llnl_unit.empty())
                {
                    std::ostringstream oss;
                    oss << mult;
                    return oss.str();
                }
                std::ostringstream oss;
                oss << "(" << mult << "*" << llnl_unit << ")";
                return oss.str();
            }

            // 4) not in REL vocabulary → error for parse()
            throw std::invalid_argument("Unrecognised unit token: '" + token + "'");
        }

        // =========================================================================
        //  Reverse mapping: llnl base_units string → REL unit name (no prefix)
        // =========================================================================
        //
        //  For each entry in kUnitMap, we parse the llnl string, pull out its
        //  base_units, and record the llnl to_string of those base_units as the
        //  key.  For entries with a scale prefix (e.g. "GHz"), we strip the
        //  prefix and only register the base name.  Predefined units (mil, cm,
        //  etc.) whose llnl base is "m" are registered as that base.
        //
        //  Duplicate keys (Ohm/Ohms both → "ohm") keep the first REL name.

        std::map<std::string, std::string>& get_base_unit_rel_name()
        {
            static std::map<std::string, std::string> map;
            static bool initialized = false;
            if (!initialized)
            {
                initialized = true;
                for (const auto& kv : kUnitMap)
                {
                    units::precise_unit pu = units::unit_from_string(kv.second);
                    if (units::isnan(pu)) continue;

                    units::precise_unit base(pu.base_units().clear_e_flag());
                    std::string base_key = units::to_string(base);
                    if (base_key.empty()) continue;

                    // first wins (aliases like Ohm/Ohms, meter/meters)
                    if (map.find(base_key) == map.end())
                        map[base_key] = kv.first;
                }
            }
            return map;
        }

    } // anonymous namespace

    // =========================================================================
    //  Auto-scale helpers (also used as reverse-mult prefix table)
    // =========================================================================

    namespace {

    struct ScaleInfo { double mult; const char* name; };

    const ScaleInfo kAutoScales[] = {
        {1e-18, "a"},  {1e-15, "f"},  {1e-12, "p"},  {1e-9,  "n"},
        {1e-6,  "u"},  {1e-3,  "m"},  {1.0,   ""},
        {1e3,   "K"},  {1e6,  "M"},   {1e9,  "G"},   {1e12, "T"},
    };
    const int kAutoScaleCount = sizeof(kAutoScales) / sizeof(kAutoScales[0]);

    }  // anonymous namespace

    // =========================================================================
    //  Unit implementation
    // =========================================================================

    Unit::Unit() : unit_() {}

    Unit::Unit(units::precise_unit pu) : unit_(std::move(pu)) {}

    Unit Unit::parse(const std::string& s)
    {
        if (s.empty())
            throw std::invalid_argument("Empty unit string");

        std::string converted = convert_token(s);
        units::precise_unit u = units::unit_from_string(converted);
        if (units::isnan(u))
        {
            throw std::invalid_argument("Unrecognised unit string: '" + s + "'");
        }
        return Unit(u);
    }

    double Unit::multiplier() const
    {
        return unit_.multiplier();
    }

    bool Unit::is_affine() const
    {
        return unit_.base_units().has_e_flag();
    }

    bool Unit::is_canonical() const
    {
        return !is_affine() && multiplier() == 1.0;
    }

    Unit Unit::canonicalized() const
    {
        return Unit(units::precise_unit(unit_.base_units().clear_e_flag()));
    }

    bool Unit::same_dimension(const Unit& other) const
    {
        return unit_.has_same_base(other.unit_);
    }

    bool Unit::has_dimension() const
    {
        return !same_dimension(Unit());
    }

    bool Unit::is_dimensionless() const
    {
        return is_canonical() && !has_dimension();
    }

    std::string Unit::to_string() const
    {
        // Step 1: canonicalise to base_units and get llnl string.
        units::precise_unit base(unit_.base_units().clear_e_flag());
        std::string base_key = units::to_string(base);

        // Step 2: check if the base_units string maps to a REL base name.
        auto& rel_map = get_base_unit_rel_name();
        auto rit = rel_map.find(base_key);
        if (rit == rel_map.end())
            return units::to_string(unit_);  // not in vocabulary — fallback

        const std::string& rel_base = rit->second;

        // Step 3: match the original multiplier to a kAutoScales prefix.
        double orig_mult = unit_.multiplier();
        const char* prefix = "";
        for (int i = 0; i < kAutoScaleCount; ++i) {
            if (kAutoScales[i].mult == orig_mult) {
                prefix = kAutoScales[i].name;
                break;
            }
        }
        // If multiplier doesn't match any known prefix, fallback.
        if (prefix[0] == '\0' && orig_mult != 1.0)
            return units::to_string(unit_);

        return std::string(prefix) + rel_base;
    }

    UnitScale Unit::best_display(double value) const
    {
        double mult = multiplier();
        Unit base_u = canonicalized();
        std::string base_str = base_u.to_string();

        // Don't auto-scale compound units (e.g. "m/s", "m^2").
        if (base_str.find_first_of("/*^") != std::string::npos)
            return {mult, base_str};

        double best_mult = 1.0;
        for (int i = kAutoScaleCount - 1; i >= 0; --i) {
            double m = kAutoScales[i].mult;
            if (m <= 0) continue;
            double absv = std::abs(value * mult / m);
            if (absv >= 1.0 && absv < 1000.0) {
                best_mult = m;
                break;
            }
        }

        // to_string() already includes the unit prefix (e.g. "MHz"),
        // so we must not multiply by `mult` a second time.
        if (best_mult == 1.0)
            return {1.0, to_string()};

        const char* prefix = "";
        for (int i = 0; i < kAutoScaleCount; ++i) {
            if (kAutoScales[i].mult == best_mult) { prefix = kAutoScales[i].name; break; }
        }
        std::string display_unit = std::string(prefix) + base_str;
        return {mult / best_mult, display_unit};
    }

    Unit Unit::operator*(const Unit& other) const
    {
        return Unit(units::precise_unit(unit_.base_units() * other.unit_.base_units()));
    }

    Unit Unit::operator/(const Unit& other) const
    {
        return Unit(units::precise_unit(unit_.base_units() / other.unit_.base_units()));
    }

    bool Unit::operator==(const Unit& other) const
    {
        return unit_ == other.unit_;
    }

    bool Unit::operator!=(const Unit& other) const
    {
        return !(*this == other);
    }

} // namespace xdataset
