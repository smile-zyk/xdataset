#ifndef XDATASET_VALUE_H
#define XDATASET_VALUE_H

#include "data_array.h"
#include "measurement.h"
#include "xdataset_predefine.h"
#include "unit.h"

#include <boost/variant.hpp>
#include <memory>
#include <string>
#include <vector>

namespace xdataset {

// =========================================================================
// Value -- unified value type for Measurement and DataArray
// =========================================================================
//
// Internally uses boost::variant<Measurement, shared_ptr<DataArray>>.
// Measurement stored on the stack (value semantics); DataArray uses shared_ptr (reference semantics).
//
// Metadata access (data_kind / data_type / shape / unit / rows) provided uniformly for both types.

class XDATASET_API Value {
public:
    /// Default: empty Measurement (kReal scalar 0, dimensionless)
    Value();

    /// Implicitly construct from Measurement
    Value(Measurement m);              // NOLINT

    /// Construct from DataArray (internal shared_ptr)
    explicit Value(const DataArray& da);

    // ---- Type queries -----------------------------------------------------
    bool is_meas()  const;
    bool is_array() const;

    // ---- Unwrap ---------------------------------------------------------
    const Measurement& as_meas()  const;
    const DataArray&   as_array() const;

    // ---- Metadata (unified for both types) ----------------------------------------
    DataKind           data_kind() const;
    DataType           data_type() const;
    DataShape          shape() const;
    const Unit&        unit() const;
    Index              rows() const;   // Measurement = 1, DataArray = data().size()
    Index              element_count() const;

private:
    typedef boost::variant<
        Measurement,
        std::shared_ptr<DataArray>
    > Storage;
    Storage storage_;
};

}  // namespace xdataset

#endif  // XDATASET_VALUE_H
