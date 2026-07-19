# DataArray 运算与单位（稳定版更新）

**最后修改：** 2026/07/19  
**阅读时间：** 10 分钟

本文档更新了旧版 `Variable` 运算设计。当前项目中：

- 旧概念 `Variable` 对应为 `DataArray`
- 旧概念 `CellSeries` 对应为 `DataSeries`
- 旧概念 `scalar` 对应为 `Measurement`（单个数据值，支持标量/向量/矩阵形态）

目标保持不变：支持 `DataArray` 之间的带单位运算，并支持与 `Measurement` 的运算。

---

## 目标

- `DataArray` 支持 `+ - * /`，可与另一个 `DataArray` 运算。
- `DataArray` 可与 `Measurement` 运算（`Measurement` 作为单值或单元值语义）。
- `pow(a, n)` / `a.pow(n)` 支持整数幂。
- 运算过程执行量纲校验并自动推导结果单位。
- 单位采用规范 SI 存储策略，保证算术热路径尽量零换算。

## 非目标（本阶段不做）

- 人类可读格式化器（自动选 SI 词头、偏好非 SI 显示单位）。
- 跨不同 `MultiDimensionSpec` 的 numpy 风格广播。
- 原地运算符（`+=`、`-=` 等）。

---

## 架构总览

```text
Layer 1  units_util.h/.cc    单位解析、量纲判断、规范化与组合工具
Layer 2  DataSeries          数据与单位承载层（unit_ + canonicalize）
Layer 3  DataArray           结构校验与结果组装层（对 DataSeries 运算做薄包装）
```

核心原则：

- 运算核心在 `DataSeries`（逐元素/广播/单位推导）。
- `DataArray` 负责结构兼容性检查与结果装配。
- `Measurement` 是单个值的统一数据模型，用于 `DataSeries` 单行访问与未来 `DataArray` 标量侧运算。

---

## 单位存储模型：规范 SI

### 当前稳定语义（与旧文档差异）

当前实现中，`DataSeries::set_unit(...)` 主要是**设置单位标签**；数值归一化由 `canonicalize()` / `canonicalized()` 完成。

- `set_unit(Unit)`：赋单位并进行基础合法性约束（例如字符串类型不能带有量纲单位）。
- `canonicalize()`：将单位和数值转换到规范 SI 形式。
- `canonicalized()`：返回规范化副本，不修改原对象。

这比旧版“`set_unit` 立即缩放数值”的描述更贴近当前代码状态，也更适合分阶段落地。

### 规范化规则

- 非仿射单位：值按 multiplier 缩放到 SI，单位转为 `base_units()` 对应的相干单位。
- 仿射单位（如 `degC` / `degF`）：必须使用 `units::convert`，正确处理偏移。
- 维度判断统一使用 `has_same_base` 语义（通过 `same_dimension` 封装）。

### 为什么坚持规范 SI

- 同一物理量有唯一内部表示。
- `+`/`-` 只需做同量纲检查后可直接算值。
- 显示层（Formatter）与存储层解耦。

---

## 运算语义（DataArray / DataSeries / Measurement）

### dtype 提升

| 左 \ 右 | Real | Integer | Complex | String |
|---|---|---|---|---|
| **Real**    | Real    | Real    | Complex | 报错 |
| **Integer** | Real    | Integer | Complex | 报错 |
| **Complex** | Complex | Complex | Complex | 报错 |
| **String**  | 报错   | 报错   | 报错   | 报错 |

规则：`Integer ⊂ Real ⊂ Complex`，`String` 不参与算术。

### DataKind 组合（按单个 Measurement 的形态）

| 左 \ 右 | Scalar | Vector | Matrix |
|---|---|---|---|
| **Scalar** | Scalar | Vector（广播） | Matrix（广播） |
| **Vector** | Vector（广播） | Vector（逐元素，需同宽） | 报错 |
| **Matrix** | Matrix（广播） | 报错 | Matrix（逐元素，需同形） |

### 单位推导

| 运算 | 结果单位 |
|---|---|
| `a + b`、`a - b` | 同量纲才允许；结果单位取左操作数（或按实现统一规范单位） |
| `a * b` | `multiply_dim(a, b)` |
| `a / b` | `divide_dim(a, b)` |
| `pow(a, n)` | `pow_dim(a, n)` |
| `a op Measurement` | 按 `Measurement` 的单位与形态参与上述规则 |

说明：无量纲参与加减是否放宽，建议在实现中作为可配置策略；默认建议严格同量纲。

### 结构兼容性（DataArray 层）

`DataArray op DataArray` 要求两侧 `MultiDimensionSpec` 一致（rank、每维 regular/ragged 结构、尺寸一致），否则抛异常。

---

## Layer 1：units_util（P1 基本完成）

已具备并稳定使用的能力：

- `parse_unit`
- `canonicalize`
- `is_affine`
- `same_dimension`
- `is_dimensionless`
- `multiplier_of`
- `multiply_dim` / `divide_dim` / `pow_dim`
- `unit_string`

目标：所有单位规则优先复用 `units_util`，避免在业务层散落单位细节。

---

## Layer 2：DataSeries + Measurement（P2 基本完成）

当前稳定状态：

- `DataSeries` 已持有 `unit_`。
- 已支持 `set_unit(Unit/string)`、`canonicalize()`、`canonicalized()`。
- 切片和拷贝路径保留单位信息。
- `measurement_at(...)` 可返回带单位 `Measurement`。

后续在此层继续完成：

- `DataSeries` 算术运算符（`+ - * / pow`）。
- `DataSeries` 与 `Measurement` 的广播/逐元素计算路径。
- 数值类型提升与错误信息标准化。

---

## Layer 3：DataArray 算术包装（下一阶段）

建议接口方向：

```cpp
const Unit& unit() const;  // 便捷返回 data().unit()

std::shared_ptr<DataArray> operator+(const DataArray& other) const;
std::shared_ptr<DataArray> operator-(const DataArray& other) const;
std::shared_ptr<DataArray> operator*(const DataArray& other) const;
std::shared_ptr<DataArray> operator/(const DataArray& other) const;
std::shared_ptr<DataArray> pow(int power) const;

std::shared_ptr<DataArray> operator+(const Measurement& m) const;
std::shared_ptr<DataArray> operator-(const Measurement& m) const;
std::shared_ptr<DataArray> operator*(const Measurement& m) const;
std::shared_ptr<DataArray> operator/(const Measurement& m) const;
```

包装层职责：

1. 校验 `MultiDimensionSpec` 兼容性（DataArray-DataArray）。
2. 委托到底层 `DataSeries` 运算。
3. 组装结果 `DataArray`（`name`、`kind`、`indep_datas`、`multi_dimension_spec` 按左操作数继承）。

---

## CMake 集成（已更新）

当前工程应使用：

1. `find_package(llnl-units CONFIG REQUIRED)`
2. 链接 target：`llnl-units::units`

说明：旧文档中的 `llnl-units::llnl-units` 在本仓库不是当前使用项。

---

## 实施阶段（状态更新）

| 阶段 | 状态 | 内容 |
|---|---|---|
| **P1** | 已完成 | `units_util` 能力落地 + CMake 引入 llnl-units |
| **P2** | 基本完成 | `DataSeries` 单位字段与规范化流程、单位传播链路 |
| **P3** | 进行中 | `DataSeries` 数值算术（先 Scalar 路径） |
| **P4** | 待开始 | `DataArray` 薄包装算术（DataArray-DataArray / DataArray-Measurement） |
| **P5** | 待开始 | Vector/Matrix 形态的广播与逐元素运算 |
| **P6** | 待开始 | `pow` 完整覆盖、单元测试、playground 示例 |

阶段目标不变：完整交付 DataArray 之间带单位的 `+ - * / pow` 运算，并支持 Measurement 参与运算。

---

## 已决策清单（更新版）

1. 术语统一到 `DataArray` / `DataSeries` / `Measurement`。
2. 单位粒度为每个 `DataSeries` 一个 `unit_`。
3. 单位规范化通过 `canonicalize` 完成（非立即在 `set_unit` 中缩放）。
4. 仿射单位规范化使用 `units::convert`，不能仅靠 multiplier。
5. DataArray 层保持薄包装，不承载底层逐元素算术细节。
6. 目标仍是 DataArray 之间的带单位运算，Measurement 作为单值参与者。