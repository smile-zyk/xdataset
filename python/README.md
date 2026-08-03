# xdataset Python bindings (pybind11)

Minimal, numpy-aligned pybind11 bindings for the xdataset core value types.
Only the three in-memory data types are exported (plus small support types):

```
xdataset
├── Measurement   # scalar | vector | matrix value + unit
├── DataArray     # a variable with data (numpy/xarray-like)
├── Value         # unified Measurement-or-DataArray
└── support: Unit, DataKind / DataType / DataArrayKind, MultiIndexSelector
```

No `Dataset` container and no HDF5 / Touchstone IO are exported — this module
is purely for manipulating the value types in memory.

## numpy alignment

All three core types behave like numpy objects:

| concept        | behaviour                                                        |
|----------------|------------------------------------------------------------------|
| `.shape`       | tuple (numpy shape)                                              |
| `.ndim`        | int                                                              |
| `.dtype`       | numpy dtype (`float64`, `int32`, `complex128`, `bool`, object)   |
| `.values`      | always a numpy array (0-d for scalars); `.value` = Python scalar |
| `np.asarray(x)`| works via `__array__` (supports `dtype=` argument)               |
| indexing       | `x[0]`, `x[1:3]`, `x[1, 2]` follow numpy semantics; scalar/row  |
|                | access that keeps the unit returns a `Measurement`               |
| arithmetic     | mixes with Python scalars, numpy scalars/arrays, Measurement,    |
|                | DataArray; units flow through the result                        |
| comparisons    | scalar -> Python bool; array -> numpy bool array (`bool` dtype)  |
| `float(x)` /   | scalar conversion (scalar-only, like numpy scalars)              |
| `int(x)` / `complex(x)` |                                            |
| `len(x)` / `iter(x)`    | first-dimension length / iteration (non-scalar)       || `rank` / `dims`         | dimension structure of a DataArray                      |
| `reduce(fn)`            | generic innermost-dimension reduction (composable)      |
## Quick tour

```python
import xdataset as xd
import numpy as np

V  = xd.Unit.parse("V")
Hz = xd.Unit.parse("Hz")

# --- Measurement: numpy scalar / vector / matrix + unit ---
m = xd.Measurement.Real(3.14); m.unit = V
float(m)            # 3.14
m.shape, m.dtype    # (), float64

v = xd.Measurement.from_numpy(np.array([1., 2., 3.]), Hz)
v.values            # array([1., 2., 3.])
v[0]                # Measurement(1 Hz) — keeps unit
v[0:2]              # array([1., 2.])
v + 1.0             # array([2., 3., 4.])
v > 1.5             # array([False,  True,  True])
np.asarray(v)       # array([1., 2., 3.])

# --- DataArray: variable with data ---
da = xd.DataArray.from_numpy(np.linspace(0, 10, 6), V)
da.values, da.shape, da.unit     # data, (6,), V
da[1:4]                          # array([2., 4., 6.])
da.row(2)                        # Measurement(4 V)
(da * 2.0 + 1.0).values          # arithmetic keeps the unit

# vector-cell DataArray (2-D input -> each row is a vector cell)
vec = xd.DataArray.from_numpy(np.arange(12.).reshape(3, 4))
vec.values                       # (3, 4) array
vec.isel([0, 2]).values          # rows 0 and 2

# innermost-dimension reduction (min/max lower the rank by one each call,
# collapsing to a single-value Independent when no dimension remains)
r = xd.DataArray.from_numpy(np.array([3., 1., 4., 2.]), V)
r.min().values                   # array([1.])  (Independent [R(1)], unit V)
r.max().values                   # array([4.])
c = xd.DataArray.from_numpy(np.array([-3+0j, 1+0j, 2+0j, -1+0j]))
c.min().values                   # array([1.+0.j])  (complex compares by |z|)
# note: min/max only support scalar data; vector/matrix cells raise RuntimeError

# --- generic reduction: reduce(fn) is the composable primitive ---
# min() == reduce(np.min), max() == reduce(np.max); any Python fn works:
r.reduce(np.mean).values         # array([2.5])
r.reduce(lambda g: g.max() - g.min()).values   # custom range

# --- building a multi-dim Dependent DataArray in Python ---
x = xd.DataArray.from_numpy(np.array([10., 20.]))        # Independent [R(2)]
y = xd.DataArray.from_numpy(np.array([1., 2., 3.]))      # Independent [R(3)]
z = xd.DataArray.create_dependent(np.array([100., 101., 102., 103., 104., 105.]),
                                  {"x": x, "y": y}, V)  # Dependent [R(2),R(3)]
z.rank, z.dims                     # 2, [2, 3]
z.reduce(np.min).values            # array([100., 103.])  (min over y per x)
z.reduce(np.min).rank              # 1 (Dependent [R(2)])
z.reduce(np.min).reduce(np.min).values   # array([100.]) (Independent [R(1)])

# --- Value: unified Measurement-or-DataArray ---
v1 = xd.Value.Real(10.0, V) + xd.Value.Real(5.0, V)
v1.value, v1.unit                # 15.0, V
a = xd.Value.ArrayReal([1., 2., 3.], V)
a.values, a > 1.5                # array([1.,2.,3.]), array([False, True, True])

# Build a Value from any numpy data (scalar -> Measurement-backed,
# array -> DataArray-backed). This is the round-trip key for writing
# "vector<Value> -> Value" computations in Python:
v = xd.Value.from_numpy(np.array([1., 2., 3.]), V)
```

## Writing a `vector<Value> -> Value` function

In Python a C++ `std::vector<Value>` is just a `list[Value]`. Combine
`.values` (numpy array), numpy operations, and `Value.from_numpy`
(numpy data + unit -> Value) to wrap any numpy pipeline:

```python
def add_one_to_first(params):
    """Add 1 to the data of the first Value, keep its unit."""
    first = params[0]
    return xd.Value.from_numpy(first.values + 1.0, first.unit)

def mean_of_params(params):
    """Element-wise mean of all parameters, unit of the first."""
    return xd.Value.from_numpy(np.stack([p.values for p in params]).mean(axis=0),
                               params[0].unit)

out = add_one_to_first([xd.Value.from_numpy(np.array([1., 2., 3.]), V)])
out.values                       # array([2., 3., 4.])
```

## Building

Requires the MSYS2/MinGW toolchain (the same one used by the `mingw-config`
preset) and the MSYS2 Python with `pybind11` and `numpy`:

```sh
pacman -S mingw-w64-x86_64-python-pybind11 mingw-w64-x86_64-python-numpy
```

The CMake configure step locates pybind11 through the Python module itself
(`python -m pybind11 --cmakedir`). On Windows the interpreter defaults to
`C:/msys64/mingw64/bin/python.exe`; override with
`-DPython3_EXECUTABLE=<path>`.

```sh
cmake --preset mingw-config -DBUILD_XDATASET_PYTHON=ON
cmake --build build --config Debug --target xdataset_pybind
```

Produces `build/Debug/xdataset.pyd` (importable name guaranteed via
`OUTPUT_NAME xdataset`, `PREFIX ""`, `SUFFIX ".pyd"`). Run:

```sh
cd build/Debug
C:/msys64/mingw64/bin/python.exe -c "import xdataset as xd; print(xd.Unit.parse('GHz'))"
```

## Demo

```sh
C:/msys64/mingw64/bin/python.exe python/demo.py
```

## Notes / library quirks

- `Measurement.Real(...)` / `Integer(...)` etc. do not take a unit (matches
  the C++ factories); assign `m.unit = ...` afterwards. `Value.*` factories
  accept an optional unit.
- `Measurement.from_numpy` / `DataArray.from_numpy` / `Value.from_numpy` map
  0-d -> scalar, 1-D -> vector cell, 2-D -> matrix cell; strings use nested
  lists / object arrays.
- Comparison results are stored internally as integers by the C++ library;
  the bindings convert them to numpy `bool` for numpy alignment.
- `DataArray.at(selectors)` only applies to vector/matrix cell data; use
  `DataArray.isel(rows)` / `select(...)` for row selection.
