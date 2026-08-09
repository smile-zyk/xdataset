#include "unit_registry.h"

#include <stdexcept>

namespace xdataset
{

// =========================================================================
//  UnitRegistry implementation
// =========================================================================

UnitRegistry::UnitRegistry()
{
    // Populate the scale-prefix table (name → factor).
    scale_map_["T"] = 1e12;
    scale_map_["G"] = 1e9;
    scale_map_["M"] = 1e6;
    scale_map_["K"] = 1e3;
    scale_map_["k"] = 1e3;
    scale_map_["_"] = 1.0;
    scale_map_["m"] = 1e-3;
    scale_map_["u"] = 1e-6;
    scale_map_["n"] = 1e-9;
    scale_map_["p"] = 1e-12;
    scale_map_["f"] = 1e-15;
    scale_map_["a"] = 1e-18;

    // ===============================================================
    //  Type A: scalable base units
    // ===============================================================

    register_base("meter", UnitData(1,0,0,0,0,0,0));           // meter
    register_alias("meters", "meter");
    register_alias("metre",  "meter");
    register_alias("metres", "meter");

    register_base("sec", UnitData(0,0,1,0,0,0,0));             // second

    register_base("Hz",  UnitData(0,0,-1,0,0,0,0));            // Hertz = s⁻¹
    register_base("Ohm", UnitData(2,1,-3,-2,0,0,0));           // Ohm = kg·m²·s⁻³·A⁻²
    register_alias("Ohms", "Ohm");
    register_base("S",   UnitData(-2,-1,3,2,0,0,0));           // Siemens = kg⁻¹·m⁻²·s³·A²
    register_base("F",   UnitData(-2,-1,4,2,0,0,0));           // Farad = kg⁻¹·m⁻²·s⁴·A²
    register_base("H",   UnitData(2,1,-2,-2,0,0,0));           // Henry = kg·m²·s⁻²·A⁻²
    register_base("V",   UnitData(2,1,-3,-1,0,0,0));           // Volt = kg·m²·s⁻³·A⁻¹
    register_base("A",   UnitData(0,0,0,1,0,0,0));             // Ampere
    register_base("W",   UnitData(2,1,-3,0,0,0,0));            // Watt = kg·m²·s⁻³

    register_base("C",   UnitData(0,0,1,1,0,0,0));             // Coulomb = A·s
    register_base("J",   UnitData(2,1,-2,0,0,0,0));            // Joule = kg·m²·s⁻²
    register_base("N",   UnitData(1,1,-2,0,0,0,0));            // Newton = kg·m·s⁻²
    register_base("Wb",  UnitData(2,1,-2,-1,0,0,0));           // Weber = kg·m²·s⁻²·A⁻¹
    register_base("Pa",  UnitData(-1,1,-2,0,0,0,0));           // Pascal = kg·m⁻¹·s⁻²

    // ===============================================================
    //  Type B: pre-defined (non-scalable)
    // ===============================================================

    register_predef("cm",   1e-2,            UnitData(1,0,0,0,0,0,0));    // centimeter
    register_predef("mil",  2.54e-5,         UnitData(1,0,0,0,0,0,0));    // mil (thousandth of an inch)
    register_predef("mils", 2.54e-5,         UnitData(1,0,0,0,0,0,0));    // plural alias
    register_predef("in",   2.54e-2,         UnitData(1,0,0,0,0,0,0));    // inch
    register_predef("ft",   12 * 2.54e-2,    UnitData(1,0,0,0,0,0,0));    // foot
    register_predef("mi",   5280 * 12 * 2.54e-2, UnitData(1,0,0,0,0,0,0)); // mile
    register_predef("nmi",  1852.0,          UnitData(1,0,0,0,0,0,0));    // nautical mile
    register_predef("PHz",  1e15,            UnitData(0,0,-1,0,0,0,0));   // petahertz
    register_predef("dB",   1.0,             UnitData(0,0,0,0,0,0,0));    // decibel (dimensionless)
}

void UnitRegistry::register_base(const std::string& name, const UnitData& dim)
{
    base_map_[name] = dim;
}

void UnitRegistry::register_alias(const std::string& alias,
                                  const std::string& base_name)
{
    alias_map_[alias] = base_name;
}

void UnitRegistry::register_predef(const std::string& name,
                                   double mult,
                                   const UnitData& dim)
{
    PredefEntry e;
    e.mult = mult;
    e.dim  = dim;
    predef_map_[name] = e;
}

const UnitData* UnitRegistry::lookup_base(const std::string& name) const
{
    // 1) direct match
    std::map<std::string, UnitData>::const_iterator it = base_map_.find(name);
    if (it != base_map_.end()) return &it->second;

    // 2) alias -> canonical -> dim
    std::map<std::string, std::string>::const_iterator ait =
        alias_map_.find(name);
    if (ait != alias_map_.end()) {
        it = base_map_.find(ait->second);
        if (it != base_map_.end()) return &it->second;
    }
    return 0;
}

const UnitRegistry::PredefEntry*
UnitRegistry::lookup_predef(const std::string& name) const
{
    std::map<std::string, PredefEntry>::const_iterator it =
        predef_map_.find(name);
    if (it != predef_map_.end()) return &it->second;
    return 0;
}

void UnitRegistry::build_reverse_map() const
{
    if (reverse_built_) return;
    for (std::map<std::string, UnitData>::const_iterator kv = base_map_.begin();
         kv != base_map_.end(); ++kv) {
        const std::string& canonical_name = kv->first;
        std::string k = kv->second.key();
        if (reverse_map_.find(k) == reverse_map_.end())
            reverse_map_[k] = canonical_name;
    }
    reverse_built_ = true;
}

const std::string* UnitRegistry::reverse_lookup(const UnitData& dim) const
{
    build_reverse_map();
    std::map<std::string, std::string>::const_iterator it =
        reverse_map_.find(dim.key());
    if (it != reverse_map_.end()) return &it->second;
    return 0;
}

const std::string* UnitRegistry::reverse_predef_lookup(double mult,
                                                       const UnitData& dim) const
{
    for (std::map<std::string, PredefEntry>::const_iterator kv = predef_map_.begin();
         kv != predef_map_.end(); ++kv) {
        if (kv->second.mult == mult && kv->second.dim == dim)
            return &kv->first;
    }
    return 0;
}

UnitRegistry::ScalePrefixMatch
UnitRegistry::try_strip_scale_prefix(const std::string& s) const
{
    ScalePrefixMatch best;
    best.found = false;
    std::size_t best_len = 0;

    for (std::map<std::string, double>::const_iterator it = scale_map_.begin();
         it != scale_map_.end(); ++it) {
        const std::string& prefix = it->first;
        const std::size_t plen = prefix.size();
        if (s.size() < plen) continue;
        if (s.compare(0, plen, prefix) != 0) continue;

        if (plen > best_len) {
            best.found     = true;
            best.factor    = it->second;
            best.remainder = s.substr(plen);
            best_len       = plen;
        }
    }
    return best;
}

const std::map<std::string, double>& UnitRegistry::scale_prefixes() const
{
    return scale_map_;
}

// =========================================================================
//  REL vocabulary
// =========================================================================

UnitRegistry& UnitRegistry::Instance()
{
    static UnitRegistry r;
    return r;
}

}  // namespace xdataset
