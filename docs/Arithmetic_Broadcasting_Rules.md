# 算术运算与广播规则

**最后修改：** 2026/07/19
**阅读时间：** 12 分钟

本文档规定 `Measurement` ↔ `Measurement`、`DataSeries` ↔ `DataSeries`、`DataSeries` ↔ `Measurement` 之间 `+ - * / pow` 的运算语义、两层广播规则、类型提升和单位推导。

---

## 1. 核心概念：两层结构

xdataset 的运算对象天然具有**两层嵌套结构**。

### 1.1 第一层：DataSeries 行结构（列级）

`DataSeries` 是一个**列**，包含 `size()` 个行。每一行是一个独立的 `Measurement`。

```text
DataSeries s (size = n)
┌──────┐
│ row 0 │  ← Measurement (shape 可以是 scalar, vector, 或 matrix)
│ row 1 │  ← Measurement
│ ...   │
│ row n-1│ ← Measurement
└──────┘
```

- 所有行共享相同的 `DataKind` 和 `data_shape()`。
- 所有行共享同一个 `Unit`（`unit()`）。

### 1.2 第二层：Measurement 元素结构（行内）

每一行内的 `Measurement` 自身也有形态：

| DataKind | Shape | 元素数 | 示例 |
|---|---|---|---|
| `Scalar` | `[]` | 1 | `3.14 m` |
| `Vector` | `[w]` | w | `[1.0, 2.0, 3.0] m` |
| `Matrix` | `[r, c]` | r×c | `[[1,2],[3,4]] m` |

### 1.3 双层运算总览

```
DataSeries op DataSeries:
  Level 1 — 行对齐: lhs.row[i] 与 rhs.row[i] 配对 (i = 0..n-1)
    要求 lhs.size() == rhs.size()
            │
            ▼
  Level 2 — Measurement 逐元素: lhs.row[i] op rhs.row[i]
    使用 §3 的规则: dtype 提升、DataKind 广播、单位推导

DataSeries op Measurement:
  Level 1 — 行广播: 单个 Measurement 广播到 DataSeries 的每一行
    无 size 约束
            │
            ▼
  Level 2 — Measurement 逐元素: lhs.row[i] op rhs_meas
    同上

Measurement op Measurement:
  只有 Level 2
```

---

## 2. 术语

| 术语 | 含义 |
|---|---|
| **dtype** | 值的数据类型：`Integer ⊂ Real ⊂ Complex`，`String` 不参与算术 |
| **DataKind** | 值的形态：`Scalar` / `Vector` / `Matrix` |
| **行内 Shape** | `Vector`: `[width]`；`Matrix`: `[rows, cols]`；`Scalar`: `[]` |
| **行数（size）** | `DataSeries::size()` |
| **广播** | 当一侧为 Scalar 时，值扩展到另一侧的每个元素 |
| **规范 SI** | `multiplier() == 1.0`、非仿射（e_flag 清除）的相干 SI 单位 |

---

## 3. Level 2 规则：Measurement ↔ Measurement

所有运算的最小单元。DataSeries 的每一行最终都委托至此。

### 3.1 前置：规范化

运算前，操作数通过 `canonicalized()` 转换为规范 SI 副本。原始对象不修改。

- 非仿射单位（`km`、`mm`、`psi`）→ `value *= multiplier_of(unit)`
- 仿射单位（`degC`、`degF`）→ `units::convert(value, src, dst)` 处理偏移
- Integer 类型需缩放时自动提升为 Real

### 3.2 dtype 提升

```
Integer ⊂ Real ⊂ Complex
```

| 左 \ 右 | Integer | Real | Complex | String |
|---|---|---|---|---|
| **Integer** | Integer | Real | Complex | ❌ |
| **Real**    | Real | Real | Complex | ❌ |
| **Complex** | Complex | Complex | Complex | ❌ |
| **String**  | ❌ | ❌ | ❌ | ❌ |

`String` 参与任何算术直接抛 `std::invalid_argument`。

### 3.3 DataKind 广播

| 左 \ 右 | Scalar | Vector | Matrix |
|---|---|---|---|
| **Scalar** | Scalar | Vector | Matrix |
| **Vector** | Vector | Vector（同宽） | ❌ |
| **Matrix** | Matrix | ❌ | Matrix（同形） |

- 广播结果 shape = 非 Scalar 一侧的 shape。
- Vector × Vector 要求 `shape[0]` 一致，Matrix × Matrix 要求 `shape[0]` 和 `shape[1]` 均一致。
- Vector × Matrix 不允许。

### 3.4 单位推导

运算前已 canonicalize，以下 `a.unit()` 均为规范 SI。

| 运算 | 结果单位 | 额外约束 |
|---|---|---|
| `a + b` | `a.unit()` | `same_dimension(a, b)` 必须成立 |
| `a - b` | `a.unit()` | `same_dimension(a, b)` 必须成立 |
| `a * b` | `multiply_dim(a, b)` | 无 |
| `a / b` | `divide_dim(a, b)` | Integer ÷ Integer → dtype 强制 Real |
| `pow(a, b)` | 见下方 | — |

**pow 单位推导：**

| 条件 | 结果单位 |
|---|---|
| `b.kind() == Scalar` 且 `b.dtype() == Integer` | `pow_dim(a.unit(), b.as_scalar<int>())` |
| 其他 | 无量纲 `Unit()` |

**pow 特殊规则：**

1. `b` 必须无量纲（`is_dimensionless(b.unit()) == true`），否则抛异常。
2. 当 `b.kind() ≠ Scalar`（即 exponent 是 Vector 或 Matrix）时：`a` 必须自身无量纲。

   *原因：若 exponent = [2, 3]，结果需同时表示为 `u²` 和 `u³`，单一 `Unit` 无法承载。*

3. `pow` 的 DataKind 广播规则与 `+ - * /` 完全一致。

### 3.5 校验顺序

1. `lhs.dtype() ≠ String && rhs.dtype() ≠ String`
2. `promoted_kind(lhs.kind(), rhs.kind())` 合法
3. 双方 canonicalize
4. （仅 `+ -`）`same_dimension(lhs.unit(), rhs.unit())`
5. （仅 `pow`）exponent 无量纲；若 exponent 非 Scalar，base 必须无量纲
6. 逐元素计算，组装结果 Measurement + 推导单位

---

## 4. DataSeries ↔ DataSeries（Level 1 + Level 2）

### 4.1 流程

```text
validate:
  lhs.size() == rhs.size()
  promoted_kind(lhs.data_kind(), rhs.data_kind()) 合法
  双方 dtype ≠ String
  (仅 +-: same_dimension)

双方 canonicalize

for each row i:
  result.set_row(i, lhs.measurement_at(i) op rhs.measurement_at(i))
```

### 4.2 结果结构

```
result.data_kind()   = promoted_kind(lhs.data_kind(), rhs.data_kind())
result.dtype()       = promoted_dtype(lhs.dtype(), rhs.dtype())
result.data_shape()  = 非 Scalar 一侧的 shape
result.size()        = lhs.size()
result.unit()        = §3.4 规则
```

### 4.3 示例

```text
Scalar 列 + Scalar 列:
  lhs: Scalar [1.0, 2.0, 3.0]
  rhs: Scalar [4.0, 5.0, 6.0]
  → result: Scalar [5.0, 7.0, 9.0]

Scalar 列 + Vector 列:
  lhs: Scalar [10.0, 20.0]
  rhs: Vector [[1,2,3], [4,5,6]]
  → row 0: Scalar(10) + Vector([1,2,3]) → Vector([11,12,13])
  → row 1: Scalar(20) + Vector([4,5,6]) → Vector([24,25,26])
  → result: Vector [[11,12,13], [24,25,26]]

Vector 列 + Vector 列:
  lhs: Vector [[1,2], [3,4]]
  rhs: Vector [[5,6], [7,8]]
  → result: Vector [[6,8], [10,12]]
```

### 4.4 校验顺序

| 步骤 | 检查 | 失败 |
|---|---|---|
| 1 | `lhs.size() == rhs.size()` | `invalid_argument("size mismatch")` |
| 2 | 双方 dtype ≠ String | `invalid_argument("string cannot participate")` |
| 3 | `promoted_kind` 合法 | `invalid_argument("incompatible DataKind")` |
| 4 | （仅 `+ -`）`same_dimension` | `invalid_argument("unit dimension mismatch")` |
| 5 | 双方 canonicalize | — |
| 6 | 逐行调用 §3 Measurement 运算 | — |

---

## 5. DataSeries ↔ Measurement（Level 1 + Level 2）

### 5.1 流程

```text
validate:
  promoted_kind(ds.data_kind(), meas.kind()) 合法
  双方 dtype ≠ String
  (仅 +-: same_dimension)

双方 canonicalize

for each row i:
  result.set_row(i, ds.measurement_at(i) op meas)
```

两层广播同时发生：

- **Level 1**：单个 Measurement 广播到 DS 的所有行
- **Level 2**：若 meas 是 Scalar 而每行是 Vector/Matrix，Scalar 逐元素广播

### 5.2 结果结构

```
result.data_kind()   = promoted_kind(ds.data_kind(), meas.kind())
result.dtype()       = promoted_dtype(ds.dtype(), meas.dtype())
result.data_shape()  = 非 Scalar 一侧的 shape
result.size()        = ds.size()
result.unit()        = §3.4 规则
```

### 5.3 示例

```text
Scalar 列 + Scalar Measurement:
  ds:  Scalar [1.0, 2.0, 3.0]
  rhs: Scalar(10.0)
  → result: Scalar [11.0, 12.0, 13.0]

Scalar 列 + Vector Measurement:
  ds:  Scalar [1.0, 2.0]
  rhs: Vector([10, 20, 30])
  → row 0: Scalar(1.0) + Vector([10,20,30]) → Vector([11,21,31])
  → row 1: Scalar(2.0) + Vector([10,20,30]) → Vector([12,22,32])
  → result: Vector [[11,21,31], [12,22,32]]

Vector 列 + Scalar Measurement:
  ds:  Vector [[1,2], [3,4]]
  rhs: Scalar(10.0)
  → row 0: Vector([1,2]) + Scalar(10) → Vector([11,12])
  → row 1: Vector([3,4]) + Scalar(10) → Vector([13,14])
  → result: Vector [[11,12], [13,14]]
```

### 5.4 校验顺序

| 步骤 | 检查 | 失败 |
|---|---|---|
| 1 | 双方 dtype ≠ String | `invalid_argument("string cannot participate")` |
| 2 | `promoted_kind` 合法 | `invalid_argument("incompatible DataKind")` |
| 3 | （仅 `+ -`）`same_dimension` | `invalid_argument("unit dimension mismatch")` |
| 4 | 双方 canonicalize | — |
| 5 | 逐行调用 §3 Measurement 运算 | — |

---

## 6. pow 跨层规则汇总

为避免分散阅读，将 `pow` 相关的所有约束在此汇总。

### 6.1 Measurement 层

| 规则 | 说明 |
|---|---|
| exponent 无量纲 | `is_dimensionless(exp.unit())` 必须为 true |
| exponent 非 String | `exp.dtype() ≠ String` |
| base 非 String | `base.dtype() ≠ String` |
| exponent 为 Scalar Integer | 结果单位 = `pow_dim(base.unit(), int)` |
| exponent 非 Scalar Integer | 结果单位为无量纲 |
| exponent 非 Scalar | base 必须无量纲 |

### 6.2 DataSeries 层

`pow(DataSeries base, Measurement exp)`：

- Level 1: 单个 exp 广播到 base 的每一行
- Level 2: 每行执行 `pow(measurement_at(i), exp)`
- 当前 `exp` 必须为 Scalar（否则不同行可能产生不同单位）

### 6.3 示例

```cpp
// base 带单位, exponent 为 int scalar — 单位可推导
Measurement len(2.0, parse_unit("m"));
Measurement area = pow(len, Measurement(2));       // 4.0 m²

// base 带单位, exponent 为 real scalar — 结果无量纲
Measurement half = pow(len, Measurement(0.5));     // 1.414, dimensionless

// exponent 为 Vector: base 必须无量纲
Measurement base(2.0);
Measurement exps(Eigen::VectorXd{{1.0, 2.0, 3.0}});
Measurement powers = pow(base, exps);              // Vector([2, 4, 8])

// DataSeries 标量列求幂
auto s = DataSeries::CreateScalarFromVector<double>({2.0, 3.0});
s.set_unit("m");
DataSeries p = pow(s, Measurement(2));             // [4.0, 9.0] m²
```

---

## 7. 完整示例

### 7.1 Measurement ↔ Measurement

```cpp
// Scalar broadcast to Vector
Measurement s(10.0);
Eigen::VectorXd v(3); v << 1.0, 2.0, 3.0;
Measurement r = s + Measurement(v);        // Vector([11, 12, 13])

// 带单位的乘除
Measurement dist(3.0, parse_unit("m"));
Measurement time(2.0, parse_unit("s"));
Measurement speed = dist / time;           // 1.5 m/s
```

### 7.2 DataSeries ↔ DataSeries

```cpp
auto a = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
a.set_unit("m");
auto b = DataSeries::CreateScalarFromVector<double>({100.0, 200.0, 300.0});
b.set_unit("cm");

DataSeries sum  = a + b;    // [2.0, 4.0, 6.0] m
DataSeries prod = a * b;    // [1.0, 4.0, 9.0] m²
DataSeries quot = a / b;    // [1.0, 1.0, 1.0] dimensionless
```

### 7.3 DataSeries ↔ Measurement

```cpp
auto series = DataSeries::CreateScalarFromVector<double>({1.0, 2.0, 3.0});
series.set_unit("m");
Measurement offset(100.0, parse_unit("cm"));

DataSeries shifted = series + offset;                // [2.0, 3.0, 4.0] m
DataSeries scaled  = series * Measurement(2.0);       // [2.0, 4.0, 6.0] m
```

---

## 8. 函数签名参考

```cpp
// ===== dtype / DataKind 提升 =====

DTypeTag promoted_dtype(DTypeTag a, DTypeTag b);
DataKind promoted_kind(DataKind a, DataKind b);

// ===== Measurement op Measurement =====

Measurement operator+(const Measurement&, const Measurement&);
Measurement operator-(const Measurement&, const Measurement&);
Measurement operator*(const Measurement&, const Measurement&);
Measurement operator/(const Measurement&, const Measurement&);
Measurement pow(const Measurement& base, const Measurement& exponent);

// ===== Measurement 规范化 =====

Measurement Measurement::canonicalized() const;

// ===== DataSeries op DataSeries =====

DataSeries operator+(const DataSeries&, const DataSeries&);
DataSeries operator-(const DataSeries&, const DataSeries&);
DataSeries operator*(const DataSeries&, const DataSeries&);
DataSeries operator/(const DataSeries&, const DataSeries&);

// ===== DataSeries op Measurement =====

DataSeries operator+(const DataSeries&, const Measurement&);
DataSeries operator-(const DataSeries&, const Measurement&);
DataSeries operator*(const DataSeries&, const Measurement&);
DataSeries operator/(const DataSeries&, const Measurement&);
DataSeries pow(const DataSeries& base, const Measurement& exp);

// ===== DataSeries 规范化 =====

DataSeries DataSeries::canonicalized() const;
```

---

## 9. 接口规范要点

1. 所有运算返回**新对象**（non-mutating），原地运算符（`+=` 等）不在范围内。
2. `+ -` 要求同量纲，否则抛异常；`* /` 不做此约束。
3. `Integer ÷ Integer` 结果强制为 `Real`（即使 dtype 提升表预测 Integer）。
4. String 参与任何算术直接抛 `std::invalid_argument`。
5. Measurement 和 DataSeries 在运算前均 canonicalize 为**副本**，原始对象不修改。
6. Vector × Matrix 在所有层均不允许。
