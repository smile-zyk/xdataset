# Variable 运算与单位

**最后修改：** 2026/07/16
**阅读时间：** 12 分钟

本文档描述为 `Variable` 支持运算（`+`、`-`、`*`、`/`、`pow`）的设计方案，包括 Variable 之间、Variable 与标量之间的运算，以及通过 [llnl-units](https://github.com/LLNL/units) 实现运行时物理单位处理。

## 目标

- `Variable` 支持 `+ - * /`，可与另一个 `Variable` 或标量运算。
- `pow(a, n)` / `a.pow(n)` 支持整数幂（C++ 没有 `**` 运算符）。
- 数据携带物理单位，单位参与运算（量纲校验、结果单位自动推导）。
- 单位以**规范 SI 形式**存储，使运算热路径上零单位换算。

## 非目标（暂不实现）

- 人类可读的格式化器（选 SI 词头如 `1e9 Hz` → `1 GHz`，或非 SI 单位如 `6894.76 Pa` → `1 psi`）。这是后续独立的 `Formatter` 类，见 [Formatter（未来）](#formatter未来)。
- 跨不同 `MultiDimensionSpec` 的广播（numpy 风格）。第一阶段要求两个操作数的维度结构完全一致。
- 原地运算符（`+=` 等）。所有运算符返回新对象。

---

## 架构总览

```
Layer 1  units_util.h/.cc        Unit 类型别名 + 解析 / 组合 / 转换工具
Layer 2  CellSeries              拥有 unit_，承载真正的 + - * / pow 运算
Layer 3  Variable                薄包装：结构校验 → 委托 CellSeries → 组装结果
```

**核心原则：运算在 `CellSeries`，不在 `Variable`。** `CellSeries` 回答"两个数组能否逐元素合并"；`Variable` 回答"两个结构化数据集能否合并"。

### 为什么单位放在 `CellSeries`（而非 `Variable`）

- `Variable::indep_datas_` 是 `ordered_map<string, CellSeries>`。若单位放在 `Variable`，每个独立坐标列都要配一个平行的 `map<string, Unit>`，脆弱且易失同步。
- 把 `unit_` 放在 `CellSeries` 上，每个坐标列自动携带自己的单位，零额外簿记。
- 数值与单位逻辑在拥有数据的那一层统一处理，实现更内聚。
- `Block` 的 `IndependentVariableInfo` / `DependentVariableInfo` 已持有 `CellSeries`，单位随 `Block` 流转无需改动。

### 为什么 `Variable` 保持薄

`Variable` 只做结构校验和结果组装：

- 检查两个操作数的 `MultiDimensionSpec` 一致（rank、每维 regular/ragged 结构）。
- 把实际计算委托给 `data_ OP other.data_`。
- 组装结果 `Variable`：`name` = 表达式形式（如 `"(a + b)"`），`kind` / `indep_datas_` / `multi_dimension_spec_` 继承自**左操作数**，`data_` = 计算出的 `CellSeries`。

---

## 单位存储模型：规范 SI

### 规则

`CellSeries::unit_` **始终**是 `multiplier == 1` 的相干 SI 单位，即 `precise_unit(u.base_units())`。

通过 `set_unit(u)` 赋单位时：

1. 单位的 multiplier 被吸收进存储数值：`stored_value = original_value * u.multiplier()`。
2. `unit_` 设为 `precise_unit(u.base_units())`（剥离 multiplier）。

此后，给定物理量的存储表示唯一，与输入单位无关。`5 ft` 与 `1.524 m` 存储完全相同（数值 `[1.524, ...]`，`unit_ = m`）。

### 为什么用规范 SI

- **热路径零换算。** 因为每个 `CellSeries` 都以相干 SI 基本尺度存值，`a + b` 只需 `a.unit_.base_units() == b.unit_.base_units()` 然后直接加数值。无运行时缩放因子，无逐元素乘法。
- **表示唯一。** 同一物理量只有一种存储形式，相等、比较、序列化都无歧义。
- **关注点分离。** "如何存储"（规范 SI）与"如何显示"（词头、非 SI 单位）解耦，后者是 `Formatter` 的职责。

### 连带后果

- `set_unit(u)` 主要用于在录入时**声明**原始（当前无量纲）数据的单位（如读 CSV 后调 `set_unit("m/s")`）。数值当场被重缩放到 SI。
- 对已有单位的 `CellSeries` 调 `set_unit`（同量纲）是 no-op（值已在 SI）。跨量纲则抛异常。
- **不存在**返回非 SI 单位 `CellSeries` 的 `convert_to(unit)`——按设计这种东西不存在。非 SI 数值通过值提取方法获取（如 `values_as("ft")`），它把存储的 SI 值除以目标单位的 multiplier。

### 仿射单位（degC / degF）

带偏移的温度单位（degC、degF）是**仿射**的，非纯乘法关系：`5 °C ≠ 5 * 1 K`，而是 `5 + 273.15 K`。因此 `set_unit` 规范化时**不能**用 `value * multiplier`（对 degC，`multiplier == 1`，会得到错误的 `5` 而非 `278.15`）。

**决策：** `set_unit` 接受任何单位输入（包括 degC/degF）。规范化时统一用 `units::convert(value, input_unit, canonical_SI)` 而非 `value * multiplier`——`convert` 正确处理仿射偏移。实测：`convert(5, degC, K) == 278.15`，`convert(32, degF, K) == 273.15`，全部正确。存储后 `unit_` 为 `K`（SI 基本单位），数值为 Kelvin 尺度。

**注意：** 仿射单位在 llnl-units 中通过 e_flag 区分（degC 的 `base_units()` 带 e_flag，K 不带），因此 `degC.base_units() != K.base_units()`。但 `has_same_base` 忽略 flag，`degC.has_same_base(K) == true`。所以 `same_dimension` 必须用 `has_same_base` 而非 `base_units() ==`（详见 [Layer 1](#layer-1units_util)）。

### 命名导出单位

`precise_unit::base_units()` 会把命名 SI 导出单位折叠成基本单位组合：

| 输入 | `base_units()` 表示 |
|---|---|
| `Hz` | `s^-1` |
| `Pa` | `kg·m^-1·s^-2` |
| `N`  | `kg·m·s^-2` |
| `J`  | `kg·m^2·s^-2` |
| `W`  | `kg·m^2·s^-3` |

**名字**不存储在 `precise_unit` 中（它只有 `multiplier_` + `base_units_` + `commodity_`）。但 `units::to_string(unit)` 会反查 `units_conversion_maps.hpp`，对相干 SI 导出单位还原名字。实测：

| 规范单位 | `to_string` 输出 |
|---|---|
| `precise_unit(Hz.base_units())` | `Hz` ✓ |
| `precise_unit(Pa.base_units())` | `Pa` ✓ |
| `precise_unit(N.base_units())`  | `N` ✓ |
| `precise_unit(V.base_units())`  | `V` ✓ |
| ...（大多数 SI 导出单位） | 原名 ✓ |
| `precise_unit(psi.base_units())` | `Pa`（名字丢失，量纲正确） |
| `precise_unit(km.base_units())`  | `m`（词头丢失，符合预期） |

因此对相干 SI 导出单位，`to_string` **免费还原名字**。只有非相干/带词头单位（psi、atm、km、GHz）在规范化后丢失显示名——这是正确的，因为它们的 multiplier 已被吸收进数值，`Formatter` 负责重新选择合适的显示单位。

---

## 运算语义

### 结果的 `kind`、`indep_datas_`、`multi_dimension_spec_`

全部继承自**左操作数**。这保持一致：结果定义在左操作数的网格上，使用左操作数的坐标列。对 independent-kind 的结果调 `set_name` 会重命名结果自身 `indep_datas_` 副本的最后一列坐标（安全——不影响原 `Variable`）。

### dtype 提升

| 左 \ 右 | Real | Integer | Complex | String |
|---|---|---|---|---|
| **Real**    | Real    | Real    | Complex | 报错 |
| **Integer** | Real    | Integer | Complex | 报错 |
| **Complex** | Complex | Complex | Complex | 报错 |
| **String**  | 报错   | 报错   | 报错   | 报错 |

规则：`Integer ⊂ Real ⊂ Complex`，提升到更宽类型。`String` 不参与算术运算。

### CellKind 组合

| 左 \ 右 | Scalar | Vector | Matrix |
|---|---|---|---|
| **Scalar** | Scalar | Vector（广播） | Matrix（广播） |
| **Vector** | Vector（广播） | Vector（逐元素，需同宽） | 报错 |
| **Matrix** | Matrix（广播） | 报错 | Matrix（逐元素，需同形） |

Scalar ↔ Vector/Matrix：标量广播到每个元素。Vector/Vector 与 Matrix/Matrix 要求形状一致，逐元素运算。

### 单位推导

| 运算 | 结果单位 |
|---|---|
| `a + b`、`a - b` | 要求同量纲（`a.unit_.base_units() == b.unit_.base_units()`），结果取左操作数单位，数值直接加减无缩放。**例外**：若其中一方无量纲（`base_units()` 为空），按 scalar 策略处理——允许运算，结果取有量纲方的单位，数值直接加减。这为易用性放宽：`a(m) + b(无量纲)` 等价于 `a + scalar`。 |
| `a * b` | `precise_unit(a.base ⊗ b.base)` |
| `a / b` | `precise_unit(a.base ⊘ b.base)` |
| `pow(a, n)` | `precise_unit(a.base^n)` |
| `a * scalar`、`scalar * a` | `a.unit_`（标量无量纲） |
| `a / scalar` | `a.unit_` |
| `a + scalar`、`a - scalar` | `a.unit_`（标量视为无量纲；有量纲量与裸数相加物理上存疑，但为易用性允许） |

因为存储是规范 SI，`+`/`-` 永远不需要重缩放——通过 `base_units()` 相等性做同量纲检查即可。

### 维度结构兼容性

`Variable op Variable` 时，两个操作数必须有**完全相同**的 `MultiDimensionSpec`（相同 rank、相同每维 regular/ragged 结构和尺寸）。否则抛异常。跨不同结构的广播是未来增强。

---

## Layer 1：`units_util`

llnl-units 的薄包装，隔离头文件依赖，提供规范 SI 原语。

```
namespace xdataset {
using Unit = units::precise_unit;

Unit parse_unit(const std::string& s);          // unit_from_string；失败返回 invalid
Unit canonicalize(const Unit& u);               // 规范 SI 单位（见下注）
bool  same_dimension(const Unit& a, const Unit& b);  // a.has_same_base(b) — 不能用 base_units()==，仿射单位 e_flag 不同
double multiplier_of(const Unit& u);            // u.multiplier()（仅非仿射单位用）
Unit  multiply_dim(const Unit& a, const Unit& b);
Unit  divide_dim(const Unit& a, const Unit& b);
Unit  pow_dim(const Unit& a, int n);
std::string unit_string(const Unit& u);         // units::to_string(u) — 还原 SI 导出名
}
```

**注（`canonicalize` 与仿射单位）：** 对非仿射单位，`canonicalize(u) = precise_unit(u.base_units())`，规范化时数值乘以 `u.multiplier()`。对仿射单位（degC/degF，带 e_flag），`base_units()` 会保留 e_flag 导致与 K 的 `base_units()` 不等，且 `multiplier == 1` 无法表达偏移。因此 `set_unit` 的规范化实现必须区分：
- 非仿射：`stored_value = value * u.multiplier()`，`unit_ = precise_unit(u.base_units().clear_e_flag())`（清掉 e_flag，得到纯 K）。
- 仿射：`stored_value = units::convert(value, u, canonical_target)`，`unit_ = precise_unit(u.base_units().clear_e_flag())`。其中 `canonical_target` 是同量纲的 SI 基本单位（degC/degF → K）。

`same_dimension` 用 `has_same_base`（忽略 e_flag/i_flag），所以 `same_dimension(degC, K) == true`，`same_dimension(m, Hz) == false`。两个规范存储的 Series（都已清 e_flag）比较时 `has_same_base` 和 `base_units() ==` 等价，但统一用 `has_same_base` 更安全。

---

## Layer 2：`CellSeries` 新增

`CellSeries` 新增成员和方法：

```
units::precise_unit unit_;   // 默认：无量纲

const Unit& unit() const;
void set_unit(const Unit& u);          // 数值乘以 u.multiplier()，存规范 base
void set_unit(const std::string& s);   // 解析后 set_unit
std::vector<double> values_as(const std::string& unit) const;  // 按非 SI 单位提取数值

CellSeries operator+(const CellSeries& other) const;
CellSeries operator-(const CellSeries& other) const;
CellSeries operator*(const CellSeries& other) const;
CellSeries operator/(const CellSeries& other) const;
CellSeries pow(int power) const;

template <typename T> CellSeries operator+(T scalar) const;
template <typename T> CellSeries operator-(T scalar) const;
template <typename T> CellSeries operator*(T scalar) const;
template <typename T> CellSeries operator/(T scalar) const;
```

每个运算符内部：
1. 提升 dtype 并校验 CellKind 兼容性。
2. 按上表推导结果单位。
3. 执行数值计算（标量用 `mutable_contiguous_data<T>()` 批量；向量/矩阵用 Eigen map）。
4. 返回新 `CellSeries`，带计算值和推导单位。

### 单位在现有操作中的传播

| 操作 | 单位处理 |
|---|---|
| 拷贝 / 移动构造、赋值 | 自动复制（unit_ 是成员） |
| `iloc` / `head` / `tail` / `at` / `select` 切片 | 结果继承源 `unit_` |
| `append_from` / `assign_from` | 不做强校验；destination 保留自己的单位（行拷贝语义；单位预期一致） |
| `append(Cell)` | `Cell` 自带 `unit_`；要求与 Series 同量纲（或 `Cell` 无量纲），值按 Series 单位追加 |
| `cell_at(i)` → `Cell` | `Cell` 继承 Series 的 `unit_`（切片即带单位） |
| `clone()` | 复制单位 |

`Cell`（单值）**带有** `unit_` 字段。因为 `Cell` 一般作为 `CellSeries` 的切片使用，切片时直接继承 Series 的单位最自然——取出的就是"带单位的值"，无需另外查 Series。`Cell` 的静态构造方法（`Scalar`/`Vector`/`Matrix`）默认产生无量纲 `Cell`；从 `CellSeries` 切片（`cell_at`）时复制 Series 的 `unit_`。这意味着 `Cell` 的头文件需要 include llnl-units。

---

## Layer 3：`Variable` 新增

```
const Unit& unit() const;   // 便捷：返回 data_.unit()

std::shared_ptr<Variable> operator+(const Variable& other) const;
std::shared_ptr<Variable> operator-(const Variable& other) const;
std::shared_ptr<Variable> operator*(const Variable& other) const;
std::shared_ptr<Variable> operator/(const Variable& other) const;
std::shared_ptr<Variable> pow(int power) const;

template <typename T> std::shared_ptr<Variable> operator*(T scalar) const;
template <typename T> std::shared_ptr<Variable> operator/(T scalar) const;
// + 标量 * Variable 的友元运算符
```

每个运算符：
1. 断言两个操作数的 `multi_dimension_spec_` 相等。
2. 委托：`CellSeries result_data = data_ OP other.data_;`
3. 组装结果 `Variable`：`name` = `"(a + b)"` 等，`kind` / `indep_datas_` / `multi_dimension_spec_` 取自左操作数，`data_` = `result_data`。

---

## Formatter（未来）

独立类（不属于 `CellSeries` / `Variable`），负责把规范 SI 的 `(value, unit)` 对渲染成人类可读字符串。

**llnl-units 免费提供的：**
- `to_string(unit)` 还原相干 SI 导出单位名（`Hz`、`Pa`、`N`...）。
- `unit_from_string` 解析带词头 / 非 SI 单位（用于用户偏好）。
- `convert(value, from, to)` 提供非 SI 目标单位的换算因子。

**llnl-units 不做（Formatter 必须自己实现）的：**
- 为**数值量级**选 SI 词头。实测 `to_string(precise_measurement(1e9, Hz_canonical))` 输出 `1000000000 Hz`，**不是** `1 GHz`。llnl-units 把数值和单位名独立处理，不选词头。

`Formatter` 核心算法（约 30 行）：

```
输入: value (SI), unit (规范 SI)
1. 若 value == 0: 返回 "0 " + to_string(unit)
2. exponent = floor(log10(abs(value)))
3. prefix_exp = round(exponent / 3) * 3      // 对齐到最近的 SI 词头步长
4. 若 prefix_exp 超出 [-24, 24]: 退回科学记法
5. mantissa = value / 10^prefix_exp
6. 返回 format(mantissa) + " " + prefix_string(prefix_exp) + to_string(unit)
   例: 1e9, Hz -> 1e9/1e9=1, "G"+"Hz" -> "1 GHz"
```

`Formatter` 后续关注点：
- 按领域的非 SI 目标单位（压力 → psi，速度 → mph）。
- 仿射温度显示（K → °C 需减 273.15，不是词头）。
- 用户指定目标单位和精度。

---

## CMake 集成

`CMakeLists.txt` 需要：
1. `find_package(llnl-units CONFIG REQUIRED)`
2. 在 `xdataset_test` 和 `xdataset_playground` 上 `target_link_libraries(... llnl-units::llnl-units)`。
3. 把 `src/units_util.cc` 加入两个可执行文件的源列表。

注意：vcpkg 安装的库名为 `units`（lib 文件 `libunits.dll.a`，DLL `libunits.dll`），llnl-units 导出的 CMake target 是 `llnl-units::llnl-units`。

---

## 实施阶段

| 阶段 | 内容 |
|---|---|
| **P1** | `units_util.h/.cc`（解析 / 规范化 / 组合 / 转换工具）+ CMake 链接 llnl-units |
| **P2** | `CellSeries` 加 `unit_` 字段 + `unit()` / `set_unit()` / `values_as()`；确保所有切片/拷贝路径传播单位 |
| **P3** | `CellSeries` 运算符：先做 Scalar × Scalar（Real / Integer / Complex 提升） |
| **P4** | `Variable` 薄包装运算符（结构校验 + 委托）；Scalar 数据端到端跑通 |
| **P5** | `CellSeries` Vector / Matrix 广播与逐元素运算 |
| **P6** | `pow` + 单位测试 + playground 示例 |

P1–P4 交付标量 `Variable` 数据的完整 `+ - * /` 带单位运算。P5 扩展到向量/矩阵 cell。P6 完成全部功能面。

---

## 已决策清单

1. **结果 `kind`**：继承自左操作数（不强制 `kDependent`）。
2. **单位粒度**：每个 `CellSeries` 一个 `unit_`（非每个 `Cell`）。`CellSeries` 表示一个物理量 × N 行，所有行同单位是天然约束。
3. **单位存储形式**：规范 SI（`multiplier == 1`，`precise_unit(base_units())`）。非 SI 值只能通过 `values_as()` 提取。
4. **仿射单位**：`set_unit` 接受任何单位（含 degC/degF）；规范化时用 `units::convert`（正确处理偏移）而非 `value * multiplier`；存储为 SI 基本单位（K）。`same_dimension` 用 `has_same_base`（忽略 e_flag）。
5. **`append_from` 单位校验**：不强校验；destination 保留自己的单位（行拷贝语义）。
6. **量级美化**：llnl-units 不提供；推迟到未来的 `Formatter` 类。
