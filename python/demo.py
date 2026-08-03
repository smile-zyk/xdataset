# =============================================================================
# demo.py — exercise the numpy-aligned pybind11 bindings for xdataset.
#
# Only three core types are exported: Measurement, DataArray, Value
# (plus the Unit support type). No Dataset / HDF5 / Touchstone IO.
#
# Run with the MSYS2 Python:
#   C:\msys64\mingw64\bin\python.exe python/demo.py
# =============================================================================
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _cand in ("build/Debug", "build/Release", "build"):
    _p = os.path.join(_REPO, _cand)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import xdataset as xd

V = xd.Unit.parse("V")
Hz = xd.Unit.parse("Hz")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Measurement — numpy scalar / vector / matrix + unit
# ---------------------------------------------------------------------------
section("Measurement (numpy-aligned)")
m = xd.Measurement.Real(3.14)
m.unit = V
print("m =", m)
print("  float(m)   =", float(m))
print("  int(m)     =", int(m))
print("  shape/dtype/ndim =", m.shape, m.dtype, m.ndim)
print("  values     =", m.values, m.values.dtype)
print("  np.asarray =", np.asarray(m))

v = xd.Measurement.from_numpy(np.array([1.0, 2.0, 3.0]), Hz)
print("v =", v)
print("  shape/dtype       =", v.shape, v.dtype)
print("  v.values          =", v.values)
print("  v[0], v[1], v[0:2] =", v[0], v[1], v[0:2])
print("  v[0] keeps unit   =", v[0].unit)
print("  list(v)           =", list(v))

# arithmetic mixes with Python scalars and numpy arrays
print("v + 1.0        =", (v + 1.0).values)
print("2 * v          =", (2 * v).values)
print("v + np.array   =", (v + np.array([0.1, 0.2, 0.3])).values)
print("v * 2 V        =", (v * 2.0).values, (v * 2.0).unit)

# comparisons return numpy bool for non-scalars, Python bool for scalars
print("v > 1.5        =", v > 1.5)
print("m > 1.0        =", m > 1.0, type(m > 1.0))
print("np.asarray(v > 1.5).dtype =", np.asarray(v > 1.5).dtype)

mat = xd.Measurement.from_numpy(np.arange(6.0).reshape(2, 3))
print("matrix:", mat.values, "| shape:", mat.shape, "| mat[1, 2] =", mat[1, 2])

# ---------------------------------------------------------------------------
# 2. DataArray — variable with values (numpy-aligned)
# ---------------------------------------------------------------------------
section("DataArray (numpy-aligned)")
da = xd.DataArray.from_numpy(np.linspace(0, 10, 6), V)
print("da =", da)
print("  values      =", da.values)
print("  shape/ndim/dtype =", da.shape, da.ndim, da.dtype)
print("  unit        =", da.unit)
print("  np.asarray  =", np.asarray(da))
print("  da[1:4]     =", da[1:4])
print("  da.row(2)   =", da.row(2), "| unit:", da.row(2).unit)

# arithmetic returns DataArray (units flow through)
one_v = xd.Measurement.Real(1.0)
one_v.unit = V
da2 = da * 2.0 + one_v
print("da*2+1 V     =", da2.values, "| unit:", da2.unit)

# comparisons return numpy bool array
print("da > 5        =", da > 5, "| dtype:", (da > 5).dtype)

# vector-cell DataArray
vec = xd.DataArray.from_numpy(np.arange(12.0).reshape(3, 4))
print("vec-cell values:\n", vec.values)
print("  shape:", vec.shape, "| isel([0,2]):\n", vec.isel([0, 2]).values)

# innermost-dimension reduction: min()/max() lower the rank by one each call,
# collapsing to a single-value Independent when no dimension remains
red = xd.DataArray.from_numpy(np.array([3.0, 1.0, 4.0, 2.0]), V)
mn, mx = red.min(), red.max()
print("reduce [R(4)]:  min ->", mn.values, "| max ->", mx.values,
      "| kind:", mn.kind, "| shape:", mn.shape, "| unit:", mn.unit)
print("  min is idempotent at the bottom:", red.min().min().values)
cplx = xd.DataArray.from_numpy(np.array([-3 + 0j, 1 + 0j, 2 + 0j, -1 + 0j]))
print("  complex compares by magnitude: min ->", cplx.min().values, "| max ->", cplx.max().values)

# min/max are just reduce(np.min)/reduce(np.max); reduce takes ANY Python fn
print("  reduce(np.min) == min():", np.allclose(red.reduce(np.min).values, red.min().values))
print("  reduce(np.max) == max():", np.allclose(red.reduce(np.max).values, red.max().values))
print("  reduce(np.mean):", red.reduce(np.mean).values)

# multi-dim Dependent built in Python: x=R(2), y=R(3), data = 100..105
x_coord = xd.DataArray.from_numpy(np.array([10.0, 20.0]))
y_coord = xd.DataArray.from_numpy(np.array([1.0, 2.0, 3.0]))
z_dep = xd.DataArray.create_dependent(np.array([100., 101., 102., 103., 104., 105.]),
                                      {"x": x_coord, "y": y_coord}, V)
print("dependent [R(2),R(3)]: rank =", z_dep.rank, "| dims =", z_dep.dims)
print("  reduce(np.min) ->", z_dep.reduce(np.min).values, "| rank:", z_dep.reduce(np.min).rank)
print("  reduce(np.sum) ->", z_dep.reduce(np.sum).values)
print("  custom range   ->", z_dep.reduce(lambda g: g.max() - g.min()).values)

# ---------------------------------------------------------------------------
# 3. Value — unified Measurement or DataArray
# ---------------------------------------------------------------------------
section("Value (numpy-aligned)")
va = xd.Value.Real(10.0, V) + xd.Value.Real(5.0, V)
print("Value.Real+V  =", va, "| value:", va.value, "| unit:", va.unit)
print("  bool(va)    =", bool(va))

arr_val = xd.Value.ArrayReal([1.0, 2.0, 3.0], V)
print("ArrayReal     =", arr_val)
print("  shape/dtype/unit =", arr_val.shape, arr_val.dtype, arr_val.unit)
print("  values      =", arr_val.values)
print("  np.asarray  =", np.asarray(arr_val))

# Value mixes with scalars / numpy arrays / Measurement
print("arr_val + 1   =", (arr_val + 1).values)
print("arr_val * 2   =", (arr_val * 2).values)
print("arr_val == 2  =", arr_val == 2)
print("arr_val > 1.5 =", arr_val > 1.5)

# ---------------------------------------------------------------------------
# 4. Writing a "vector<Value> -> Value" function (user pattern)
# ---------------------------------------------------------------------------
section("vector<Value> -> Value (user pattern)")
# In Python a C++ `std::vector<Value>` is just a `list[Value]`.
# Value.from_numpy(...) builds a Value back from numpy data + a unit, so any
# numpy pipeline can be wrapped as a Value -> Value computation.


def add_one_to_first(params):
    """Add 1 to the data of the first Value, keep its unit."""
    first = params[0]
    return xd.Value.from_numpy(first.values + 1.0, first.unit)


def mean_of_params(params):
    """Element-wise mean of all parameters, unit of the first."""
    return xd.Value.from_numpy(np.stack([p.values for p in params]).mean(axis=0),
                               params[0].unit)


params = [xd.Value.from_numpy(np.array([1.0, 2.0, 3.0]), V),
          xd.Value.from_numpy(np.array([4.0, 5.0, 6.0]), V),
          xd.Value.from_numpy(np.array([7.0, 8.0, 9.0]), V)]
print("params =", [p.values for p in params])
out = add_one_to_first(params)
print("add_one_to_first ->", out.values, "| unit:", out.unit, "|", type(out).__name__)
print("mean_of_params   ->", mean_of_params(params).values)

print("\nAll demo steps completed successfully.")
