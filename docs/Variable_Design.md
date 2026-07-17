# Variable 设计

**最后修改：** 2026/07/17  
**阅读时间：** 10 分钟

本文档描述新 `Variable` 类型的设计方案——一个表示**单行数据**（可以是数学标量、向量或矩阵）的值类型，不涉及额外堆分配，内置单位支持，在库中与 `DataFrame` 处于同一使用层级。

> **命名变更**：原有的多行多维 `Variable` 更名为 **`DataFrame`**（对标 pandas 概念），释放 `Variable` 这个名字给单行值类型使用，最终形成 `Block → DataFrame → CellSeries → Variable` 的清晰层级。

## 动机

当前代码中存在两个功能重叠的类型：

| 类型 | 定位 | 问题 |
|------|------|------|
| `Cell` | 单行数据（标量/向量/矩阵） | 通过 `unique_ptr<CellStorage>` + 虚函数实现类型擦除，**每次构造必然堆分配**；虽已携带 `unit_`，但整体偏重 |
| `GridField` | `GridModel` 表中的单个单元格值 | `boost::variant<double, int, complex<double>, string>`，**仅支持标量，无向量/矩阵，无单位** |

二者的核心语义重叠：都表示"一个数据点"。`Variable` 将它们统一为一个轻量值类型。

## 目标

1. **单行语义**：表示一组观测中的单条记录——可以是数学标量（0-d）、向量（1-d）或矩阵（2-d）。
2. **无额外堆分配**：不使用 `unique_ptr` + 虚函数，数据直接嵌入对象。
3. **内置单位**：每个 `Variable` 携带 `Unit`（来自 `units_util.h`）。
4. **一等公民**：与 `DataFrame`（原 `Variable`）同层级，可独立于 `CellSeries` / `DataFrame` 使用。
5. **替换 `Cell` 和 `GridField`**：统一全库的单值表示。
6. **C++11 兼容**：使用已有的 `boost::variant`（项目已依赖）。

## 非目标（暂不实现）

- 运行时可变的 `CellKind` + `DTypeTag` 任意组合下的类型安全访问——保留现有 `CellSeries::RowView` 提供直接视图访问。
- 原地算术运算符。返回新 `Variable`。
- 跨 `Variable` 的广播（numpy 风格）。第一阶段要求两个操作数的 shape 完全一致。

---

## 全库类型层级

```
Block ──▶ DataFrame ──▶ CellSeries ──▶ Variable
(数据块)   (命名多维表格)  (同构列容器)    (单行带单位值)

原名映射：
  旧 Variable     → 新 DataFrame
  旧 Cell         → 新 Variable
  旧 GridField    → 新 Variable
  旧 RowView      → 新 VariableView
  旧 ConstRowView → 新 ConstVariableView
```

---

## 架构定位

```
Layer 1  units_util.h          Unit 类型别名 + 解析 / 组合 / 转换工具
Layer 2  Variable              单行数据值类型：kind + dtype + shape + unit + variant storage
Layer 3  CellSeries            多行同构容器，内部存储连续数据；单行视图返回 Variable / VariableView
Layer 4  DataFrame             命名多维表格，委托 CellSeries 运算（原 Variable）
Layer 5  Block                 数据块，包含多个 DataFrame
```

**关键区别**：`CellSeries` 是多行的**容器**（`std::vector` 语义），`Variable` 是单行的**值**（`int` 语义）。`Variable` 可以被拷贝、移动、作为函数参数按值传递，不会触发 `CellSeries` 或 `DataFrame` 的开销。

---

## 存储模型

### 概述

```
┌──────────────────────────────────────────────┐
│  Variable                                    │
│  ┌──────────┬──────────┬──────────┬────────┐ │
│  │ kind_    │ dtype_   │ shape_   │ unit_  │ │  元数据（栈上）
│  │ CellKind │ DTypeTag │ vector<  │ Unit   │ │
│  │ (4B)     │ (4B)     │ Index>   │ (??B)  │ │
│  └──────────┴──────────┴──────────┴────────┘ │
│  ┌──────────────────────────────────────────┐ │
│  │  storage_  (boost::variant<...>)         │ │  数据（栈上嵌入）
│  │  double | int | complex<double> | string │ │
│  │  | VectorXd | VectorXi | VectorXcd      │ │
│  │  | Tensor<string,1>                     │ │
│  │  | MatrixXd | MatrixXi | MatrixXcd      │ │
│  │  | Tensor<string,2>                     │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### 与旧 `Cell` 的对比

```
旧 Cell:
  Cell ──unique_ptr──▶ CellStorage (虚基类)
                          △
              ┌───────────┼───────────┐
              │           │           │
          ScalarStorage  VectorStorage MatrixStorage
          <T>            Numeric<T>   Numeric<T>
          (内部 vector<T>) (内部 vector<T>) (内部 vector<T>)

  每次构造 Cell → 1× new CellStorage + 1× vector 扩容 = 至少 2 次堆分配
  每次访问 → 虚函数调用 (vtable dispatch)

新 Variable:
  Variable
   └── boost::variant< double, int, complex<double>, string,
                        VectorXd, VectorXi, VectorXcd, Tensor<string,1>,
                        MatrixXd, MatrixXi, MatrixXcd, Tensor<string,2> >

  标量类型 (double/int/complex/string) → 零堆分配，完全栈上
  向量/矩阵 (Eigen) → Eigen 内部 1 次堆分配（元素存储），无虚函数
  每次构造 → 无 unique_ptr new，variant 直接栈上构造
  每次访问 → 编译期确定的模板访问或 boost::apply_visitor
```

### 关键洞察

- **标量路径是零开销的**：`int`、`double` 直接存入 variant，不需要任何堆内存。
- **向量/矩阵路径**：Eigen 动态类型内部仍需堆分配元素，但省去了 `CellStorage` 的额外虚表指针和解引用。
- **`variant` 代替继承**：类型集合在编译期已知（12 种组合），因此用 `boost::variant` 替代虚基类 `CellStorage`，将 vtable dispatch 替换为 switch-case / visitor dispatch。

---

## API 设计

### 类型定义（沿用现有枚举）

```cpp
namespace xdataset {

// 复用现有枚举（来自 cell_series.h）
enum class CellKind   { kScalar, kVector, kMatrix };
enum class DTypeTag   { kReal, kInteger, kComplex, kString };
using Unit = units::precise_unit;

// ── 内部存储 variant ──────────────────────────────

// 每种 (CellKind, DTypeTag) 对应一个 variant alternative
using VariableStorageVariant = boost::variant<
    // --- CellKind::kScalar ---
    double,
    int,
    std::complex<double>,
    std::string,

    // --- CellKind::kVector ---
    Eigen::VectorXd,                          // DTypeTag::kReal
    Eigen::VectorXi,                          // DTypeTag::kInteger
    Eigen::VectorXcd,                         // DTypeTag::kComplex
    Eigen::Tensor<std::string, 1>,            // DTypeTag::kString

    // --- CellKind::kMatrix ---
    Eigen::MatrixXd,                          // DTypeTag::kReal
    Eigen::MatrixXi,                          // DTypeTag::kInteger
    Eigen::MatrixXcd,                         // DTypeTag::kComplex
    Eigen::Tensor<std::string, 2>             // DTypeTag::kString
>;

// ── 前向声明 ─────────────────────────────────────
class Variable;
class VariableView;          // 非拥有视图（指向 CellSeries 中某行）
class ConstVariableView;     // 只读视图
```

### `Variable` 公开接口

```cpp
class Variable {
public:
    // ======== 构造与赋值 ========

    // 默认构造：kReal 标量，值为 0.0，单位为空
    Variable();

    // 拷贝 / 移动
    Variable(const Variable& other);
    Variable& operator=(const Variable& other);
    Variable(Variable&& other) noexcept;
    Variable& operator=(Variable&& other) noexcept;

    // ======== 静态工厂方法 ========

    /// @{
    /// 标量（0-d）工厂
    static Variable Real(double value);
    static Variable Integer(int value);
    static Variable Complex(std::complex<double> value);
    static Variable String(const std::string& value);
    static Variable String(std::string&& value);
    /// @}

    /// @{
    /// 向量（1-d）工厂 —— 数值类型
    template <typename T>
    static typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Variable
    >::type Vector(const Eigen::Matrix<T, Eigen::Dynamic, 1>& values);

    template <typename T>
    static typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Variable
    >::type Vector(Eigen::Matrix<T, Eigen::Dynamic, 1>&& values);

    /// 向量（1-d）工厂 —— 字符串类型
    static Variable Vector(const Eigen::Tensor<std::string, 1>& values);
    static Variable Vector(Eigen::Tensor<std::string, 1>&& values);

    /// 向量零值工厂（指定宽度）
    template <typename T>
    static Variable Vector(Index width);
    /// @}

    /// @{
    /// 矩阵（2-d）工厂 —— 数值类型
    template <typename T>
    static typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Variable
    >::type Matrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& values);

    template <typename T>
    static typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Variable
    >::type Matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&& values);

    /// 矩阵（2-d）工厂 —— 字符串类型
    static Variable Matrix(const Eigen::Tensor<std::string, 2>& values);
    static Variable Matrix(Eigen::Tensor<std::string, 2>&& values);

    /// 矩阵零值工厂（指定行列数）
    template <typename T>
    static Variable Matrix(Index rows, Index cols);
    /// @}

    // ======== 类型查询 ========

    bool has_value() const;
    CellKind kind() const;
    DTypeTag dtype() const;
    std::vector<Index> shape() const;        // 标量→[], 向量→[w], 矩阵→[r,c]
    std::string dtype_name() const;

    // ======== 单位 ========

    const Unit& unit() const;
    void set_unit(const Unit& u);
    void set_unit(const std::string& s);     // 解析单位字符串

    // ======== 数据访问 ========

    /// @{
    /// 标量访问（kind_ != kScalar 时抛 std::bad_cast）
    template <typename T> T& as_scalar();
    template <typename T> const T& as_scalar() const;
    /// @}

    /// @{
    /// 向量访问（kind_ != kVector 时抛 std::bad_cast）
    template <typename T>
    typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>
    >::type as_vector();

    template <typename T>
    typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
    >::type as_vector() const;

    // 字符串向量返回 Tensor 引用
    Eigen::Tensor<std::string, 1>& as_vector_str();
    const Eigen::Tensor<std::string, 1>& as_vector_str() const;
    /// @}

    /// @{
    /// 矩阵访问（kind_ != kMatrix 时抛 std::bad_cast）
    template <typename T>
    typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    >::type as_matrix();

    template <typename T>
    typename std::enable_if<
        !std::is_same<T, std::string>::value,
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    >::type as_matrix() const;

    Eigen::Tensor<std::string, 2>& as_matrix_str();
    const Eigen::Tensor<std::string, 2>& as_matrix_str() const;
    /// @}

    // ======== 类型擦除访问（visitor 模式）=======

    /// 对内部存储应用 boost::static_visitor
    template <typename Visitor>
    typename Visitor::result_type apply_visitor(Visitor& v);

    template <typename Visitor>
    typename Visitor::result_type apply_visitor(const Visitor& v) const;

    // ======== 转为 CellSeries（单行）=======

    /// 创建一个仅包含当前值的单行 CellSeries
    CellSeries to_cell_series() const;

    // ======== 运算符 ========

    Variable operator+(const Variable& other) const;
    Variable operator-(const Variable& other) const;
    Variable operator*(const Variable& other) const;
    Variable operator/(const Variable& other) const;

    // 与裸标量运算的便捷重载
    Variable operator+(double v) const;
    Variable operator-(double v) const;
    Variable operator*(double v) const;
    Variable operator/(double v) const;

    Variable pow(int n) const;

    bool operator==(const Variable& other) const;
    bool operator!=(const Variable& other) const;

private:
    CellKind                 kind_;
    DTypeTag                 dtype_;
    std::vector<Index>       shape_;      // 冗余但加速查询（避免每次访问 variant）
    Unit                     unit_;
    VariableStorageVariant   storage_;
};
```

### `VariableView` / `ConstVariableView`（非拥有视图）

```cpp
// 非拥有视图：指向 CellSeries 中某行数据
// 不拥有存储，仅提供类型安全的访问接口

class VariableView {
public:
    VariableView(CellSeries* owner, Index idx);

    bool has_value() const;
    CellKind kind() const;
    DTypeTag dtype() const;
    std::vector<Index> shape() const;
    Index index() const;
    const Unit& unit() const;

    // 访问方法同 Variable，但直接操作 CellSeries 底层存储
    template <typename T> T& as_scalar();
    template <typename T> const T& as_scalar() const;
    // ... vector / matrix 同理

    Variable to_variable() const;  // 深拷贝为拥有型 Variable

private:
    CellSeries* owner_;
    Index idx_;
};

class ConstVariableView {
    // 同上，但所有访问返回 const 引用
};
```

---

## 与现有类型的集成计划

### 1. 替换 `Cell`

| 旧 `Cell` | 新 `Variable` |
|-----------|---------------|
| `Cell::Scalar<double>(v)` | `Variable::Real(v)` |
| `Cell::Vector<double>(vec)` | `Variable::Vector<double>(vec)` |
| `Cell::Matrix<double>(mat)` | `Variable::Matrix<double>(mat)` |
| `Cell::Scalar<double>()` | `Variable::Real(0.0)` |
| `cell.scalar<T>()` | `var.as_scalar<T>()` |
| `cell.vector<T>()` | `var.as_vector<T>()` |
| `cell.set_unit(u)` | `var.set_unit(u)` |

**迁移策略**：先引入 `Variable`，标记 `Cell` 为 deprecated（`[[deprecated]]`），逐步替换所有调用点后删除 `Cell`。

### 2. 替换 `GridField`

| 旧 `GridField` | 新 `Variable` |
|----------------|---------------|
| `boost::variant<double, int, complex<double>, string>` | `Variable`（包含 kind + dtype + 值 + 单位） |
| 无法表示向量/矩阵 | 完整支持 |

**迁移策略**：
- `GridRow::fields` 从 `vector<GridField>` 改为 `vector<Variable>`
- `GridFieldStringifyVisitor` / `GridFieldGetVisitor` 改为 `VariableStringifyVisitor` / `VariableGetVisitor`
- `GridModel` 输出可携带单位信息

### 3. `Variable`（旧）→ `DataFrame` 重命名

| 旧 | 新 |
|----|-----|
| `Variable`（多行多维） | `DataFrame` |
| `VariableCreateInfo` | `DataFrameCreateInfo` |
| `VariableKind` | `DataFrameKind` |
| `VariableDescriptor` | `DataFrameDescriptor` |
| `VariableDescriptorCreateInfo` | `DataFrameDescriptorCreateInfo` |

语义不变，仅改名。`DataFrame` 表示一个具名的、绑定独立坐标轴的、多维结构化的数据表格。

### 4. 与 `CellSeries` 的关系

```
CellSeries::row(i)         → 返回 VariableView（视图，零拷贝，用于遍历/读写）
CellSeries::at(i)          → 返回 Variable（拥有型拷贝，用于需要独立值的场景）
CellSeries::append_from(src, row)  → 内部行间拷贝，不变
CellSeries::append(var)            → 接收 Variable&& 或 const Variable&
```

- `CellSeries` 内部存储**不变**——仍然使用同构的连续存储（`ScalarStorage<T>` / `VectorStorageNumeric<T>` 等）。
- `Variable` 仅作为外部接口层的"单行值"类型。
- `VariableView` 提供零拷贝的行级访问。

### 5. 单位语义

继承现有约定：`unit_` 始终存储为规范 SI（`multiplier == 1`）。`set_unit(u)` 时 multiplier 被吸收进数值。

`Variable` 的运算符（`+ - * /`）遵循：
- 量纲校验：`same_dimension(a.unit(), b.unit())`（加/减），或推导结果单位（乘/除/pow）
- 结果 `Variable` 的单位 = 运算推导出的规范 SI 单位

---

## 使用示例

```cpp
using namespace xdataset;

// ── 构造 ──────────────────────────────────────────
Variable v1 = Variable::Real(3.14);
v1.set_unit("m/s");

Variable v2 = Variable::Vector<double>(Eigen::Vector3d(1.0, 2.0, 3.0));
v2.set_unit("V");

Variable v3 = Variable::Matrix<double>(Eigen::Matrix2d::Identity());
v3.set_unit("Hz");

// ── 类型查询 ──────────────────────────────────────
assert(v1.kind() == CellKind::kScalar);
assert(v1.dtype() == DTypeTag::kReal);
assert(v1.shape().empty());

assert(v2.kind() == CellKind::kVector);
assert(v2.shape().size() == 1 && v2.shape()[0] == 3);

// ── 数据访问 ──────────────────────────────────────
double& x = v1.as_scalar<double>();
x = 5.0;

auto v = v2.as_vector<double>();  // Eigen::Map<Eigen::VectorXd>
v(0) = 10.0;

auto m = v3.as_matrix<double>();  // Eigen::Map<Eigen::MatrixXd>

// ── 拷贝 ──────────────────────────────────────────
Variable v4 = v1;  // 深拷贝，独立于 v1
v4.as_scalar<double>() = 7.0;
assert(v1.as_scalar<double>() == 5.0);  // v1 不受影响

// ── 与 CellSeries / DataFrame 交互 ────────────────
CellSeries series = CellSeries::Scalars<double>(3, 0.0);
series.scalar_at<double>(1) = 42.0;

VariableView view = series.row(1);          // 零拷贝视图
assert(view.as_scalar<double>() == 42.0);
view.as_scalar<double>() = 99.0;            // 直接修改 CellSeries 底层数据

Variable owned = series.at(1);              // 深拷贝

// ── 运算符 ────────────────────────────────────────
Variable a = Variable::Real(10.0);
a.set_unit("m");
Variable b = Variable::Real(2.0);
b.set_unit("s");
Variable c = a / b;   // c = 5.0, unit = "m/s"
```

---

## 实现注意事项

### 1. `boost::variant` 类型映射

variant 的 which() 与 (CellKind, DTypeTag) 的对应关系：

| which() | CellKind | DTypeTag | C++ 类型 |
|---------|----------|----------|----------|
| 0 | kScalar | kReal | `double` |
| 1 | kScalar | kInteger | `int` |
| 2 | kScalar | kComplex | `std::complex<double>` |
| 3 | kScalar | kString | `std::string` |
| 4 | kVector | kReal | `Eigen::VectorXd` |
| 5 | kVector | kInteger | `Eigen::VectorXi` |
| 6 | kVector | kComplex | `Eigen::VectorXcd` |
| 7 | kVector | kString | `Eigen::Tensor<std::string, 1>` |
| 8 | kMatrix | kReal | `Eigen::MatrixXd` |
| 9 | kMatrix | kInteger | `Eigen::MatrixXi` |
| 10 | kMatrix | kComplex | `Eigen::MatrixXcd` |
| 11 | kMatrix | kString | `Eigen::Tensor<std::string, 2>` |

### 2. `shape_` 冗余但必要

虽然对于 variant 中已有的 Eigen 类型可以通过 `.rows()` / `.cols()` 获取 shape，但：
- `kind_` 查询不需要访问 variant（更快）
- 统一的 `shape()` 接口无需 visitor
- 空 `Variable` 或无值状态可仅靠 `kind_` + 空 `shape_` 判断

### 3. C++11 限制

- 使用 `boost::variant`（已依赖），不使用 C++17 `std::variant`
- 使用 `boost::apply_visitor`，不使用 C++17 `std::visit`
- `enable_if` 而非 `if constexpr`
- `noexcept` 移动构造（`boost::variant` 移动不抛异常）

### 4. 头文件组织

```
include/xdataset/variable.h        ← Variable, VariableView, ConstVariableView 定义
include/xdataset/data_frame.h      ← DataFrame（原 Variable）定义
```

`Variable` 是 header-only 模板类；`DataFrame` 有独立的 `.cc` 实现文件。

### 5. 运算符实现策略

```
Variable 运算符:
  ├── kind_/dtype_/shape_ 完全一致 → 直接 variant dispatch 逐元素运算
  ├── 标量 Variable × 一般 Variable → 广播标量到每个元素
  └── 单位校验
          ├── + / - → same_dimension()
          ├── * → multiply_dim()
          ├── / → divide_dim()
          └── pow → pow_dim()
```

---

## 迁移步骤

1. **Phase 1**：创建 `variable.h`，实现新 `Variable` + `VariableView` + `ConstVariableView`。旧 `Variable` → `DataFrame` 重命名（`data_frame.h`）。两者并存。
2. **Phase 2**：`CellSeries` 增加 `at()`（返回 `Variable`），`row()` 改为返回 `VariableView`（旧 `RowView` 标记 deprecated）。
3. **Phase 3**：`GridModel` 迁移：`GridField` → `Variable`。
4. **Phase 4**：删除 `Cell` 类、`GridField` 别名、旧 `RowView` / `ConstRowView`。

---

## 参考

- [Variable 运算与单位](Variable_Arithmetic_Units.md) — `CellSeries` + `DataFrame`（原 `Variable`）运算设计
- [Data Structures](Data_Structures.md) — DataStore / DataBlock / DataFrame 概念
- [llnl-units](https://github.com/LLNL/units) — 物理单位库
- `cell_series.h` — 现有 `Cell` / `CellSeries` / `RowView` 实现
- `grid_model.h` — 现有 `GridField` / `GridRow` / `GridModel` 实现
