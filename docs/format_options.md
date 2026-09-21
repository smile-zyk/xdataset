# Measurement 显示格式选项（FormatOptions）

`Measurement::to_string()` 的显示行为由 `xdataset::FormatOptions` 控制。
本文所有输出均为**实测值**（由 `Measurement::to_string(options)` 实际生成），非推算。

```cpp
// 3rd/xdataset/include/measurement.h
struct FormatOptions
{
    NumberFormat  number_format      = NumberFormat::kFull;          // 默认 Full
    ComplexFormat complex_format     = ComplexFormat::kRealImaginary;
    int           significant_digits = 6;                            // 钳制到 [1, 17]
    bool          show_unit          = true;
};

std::string Measurement::to_string() const;                     // 用全局默认 options
std::string Measurement::to_string(const FormatOptions&) const; // 显式指定
```

> **注意（相对历史行为的破坏性变更）**：历史实现只有一个"自动选前缀 + `%.6g`"
> 的行为（曾以 `kDefault` 表示）。该模式**已删除**，`kFull` 现在是默认值。
> 因此 `0.002 V` 默认输出 `0.002 V`（历史为 `2 mV`）。
> 需要自动选前缀的调用方请显式使用 `NumberFormat::kEngineering`。

---

## 1. NumberFormat（实数 / 整数）

| 枚举 | 含义 | 单位自动缩放 |
|---|---|---|
| `kFull` | 整数位全显示，无指数；有效位预算**只截断、不补零** | ❌ |
| `kScientific` | 科学计数法 `1.53e6` | ❌ |
| `kEngineering` | 指数强制为 3 的倍数，用 SI 前缀表达 | ✅ |
| `kHex` | 十六进制，`0x` 前缀 | ❌（且无单位后缀） |
| `kOctal` | 八进制，`0` 前缀 | ❌（且无单位后缀） |
| `kBinary` | 二进制，`0b` 前缀 | ❌（且无单位后缀） |

### 1.1 `show_unit = true`（默认）

| 值 | Full | Scientific | Engineering | Hex | Octal | Binary |
|---|---|---|---|---|---|---|
| `1530000.123 Hz` | `1530000 Hz` | `1.53e6 Hz` | `1.53 MHz` | `1530000` | `1530000` | `1530000` |
| `1000 Hz` | `1000 Hz` | `1e3 Hz` | `1 KHz` | `0x3e8` | `01750` | `0b1111101000` |
| `2.4e9 Hz` | `2400000000 Hz` | `2.4e9 Hz` | `2.4 GHz` | `0x8f0d1800` | `021703214000` | `0b10001111000011010001100000000000` |
| `50 Hz` | `50 Hz` | `5e1 Hz` | `50 Hz` | `0x32` | `062` | `0b110010` |
| `0.002 V` | `0.002 V` | `2e-3 V` | `2 mV` | `0.002` | `0.002` | `0.002` |
| `4700 Ohm` | `4700 Ohm` | `4.7e3 Ohm` | `4.7 KOhm` | `0x125c` | `011134` | `0b1001001011100` |
| `0.002`（无量纲） | `0.002` | `2e-3` | `2 m` | `0.002` | `0.002` | `0.002` |
| `1e9`（无量纲） | `1000000000` | `1e9` | `1 G` | `0x3b9aca00` | `07346545000` | `0b111011100110101100101000000000` |
| `255`（整数） | `255` | `2.55e2` | `255` | `0xff` | `0377` | `0b11111111` |
| `1.5`（实数） | `1.5` | `1.5e0` | `1.5` | `1.5` | `1.5` | `1.5` |

注意 `Full` 列**不再补零**：`1000 Hz` → `1000 Hz`（历史为 `1000.00 Hz`），
`255` → `255`（历史为 `255.000`），`1.5` → `1.5`（历史为 `1.50000`）。

### 1.2 `show_unit = false`

| 值 | Full | Scientific | Engineering | Hex | Octal | Binary |
|---|---|---|---|---|---|---|
| `1530000.123 Hz` | `1530000` | `1.53e6` | **`1.53M`** | `1530000` | `1530000` | `1530000` |
| `1000 Hz` | `1000` | `1e3` | **`1K`** | `0x3e8` | `01750` | `0b1111101000` |
| `2.4e9 Hz` | `2400000000` | `2.4e9` | **`2.4G`** | `0x8f0d1800` | `021703214000` | `0b10001111000011010001100000000000` |
| `50 Hz` | `50` | `5e1` | `50` | `0x32` | `062` | `0b110010` |
| `0.002 V` | `0.002` | `2e-3` | **`2m`** | `0.002` | `0.002` | `0.002` |
| `4700 Ohm` | `4700` | `4.7e3` | **`4.7K`** | `0x125c` | `011134` | `0b1001001011100` |
| `0.002`（无量纲） | `0.002` | `2e-3` | **`2m`** | `0.002` | `0.002` | `0.002` |
| `1e9`（无量纲） | `1000000000` | `1e9` | **`1G`** | `0x3b9aca00` | `07346545000` | `0b111011100110101100101000000000` |
| `255`（整数） | `255` | `2.55e2` | `255` | `0xff` | `0377` | `0b11111111` |
| `1.5`（实数） | `1.5` | `1.5e0` | `1.5` | `1.5` | `1.5` | `1.5` |

### 1.3 `show_unit` 的语义

- **`true`（默认）**：显示完整单位后缀（含前缀），如 `2.4 GHz`。
- **`false`**：
  - **Full / Scientific / Hex / Octal / Binary**：**裸值，且不缩放**。因为缩放是靠单位前缀表达的，
    没有单位就无法表达缩放 —— `0.002 V` 显示 `0.002` 而不是 `2`（后者会被误读为 2 伏特）。
  - **Engineering**：**例外**。10³ 步进属于**模式本身**而非单位，所以缩放照常执行，
    只保留前缀，SPICE 风格**无空格**拼接：`2.4G`、`4.7K`、`2m`。

### 1.4 无量纲值

无量纲值**同样**在 `kEngineering` 下应用 best unit display：`0.002 → 2 m`、`1e9 → 1 G`。

注意：`m`/`M`/`K`/`G` 是**无量纲前缀，不是单位**，因此不能作为"目标单位"被选中
（`FormatOptions` 里没有 `display_unit` 字段）。但在 `kEngineering` 下它们会被自动施加
——这正是 "best unit display" 的含义。

### 1.5 进制模式的回退

`kHex` / `kOctal` / `kBinary` 只能表示**精确整数**。非整数或 `|v| > 9e15` 时回退到 Full 十进制：

```
1.5   kHex  -> 1.5        （不是 0x1.8）
0.002 kHex  -> 0.002      （不是任何十六进制）
```

---

## 2. significant_digits（有效位数）

作用于 `kFull` / `kScientific` / `kEngineering`，钳制到 `[1, 17]`。
`kHex` / `kOctal` / `kBinary` 忽略它（精确整数表示）。

以 `1234.567890 Hz` 为例：

| digits | Full | Scientific | Engineering |
|---|---|---|---|
| 1 | `1235 Hz` | `1e3 Hz` | `1 KHz` |
| 3 | `1235 Hz` | `1.2e3 Hz` | `1.23 KHz` |
| 6 | `1234.57 Hz` | `1.2346e3 Hz` | `1.23457 KHz` |
| 9 | `1234.56789 Hz` | `1.2345679e3 Hz` | `1.23456789 KHz` |
| 12 | `1234.56789 Hz` | `1.23456789e3 Hz` | `1.23456789 KHz` |
| 17 | `1234.56789 Hz` | `1.23456789e3 Hz` | `1.2345678900000001 KHz` |

**`kFull` 的预算分配规则**：有效位预算**先花在整数位上**，剩余才作小数位；
并且**只截断、绝不补零**。

- `1234.567890` 有 4 个整数位 → 6 位预算剩 2 位小数 → `1234.57`
- `1530000.123` 有 7 个整数位 → 已超出 6 位预算 → `1530000`（无小数）
- `1000.0` 有 4 个整数位 → 剩 2 位预算 → 实际只有 0 位有效小数 → `1000`（不是 `1000.00`）
- `0.002` 有 0 个整数位（`int_digits = -2`）→ 预算反而更大 → `0.002`（不是 `0.00200000`）

因此 digits=12 与 digits=17 在 `1234.567890` 上给出**相同**结果 —— 多出的预算没有
可填的有效数字，不会被零填充。

---

## 3. ComplexFormat（复数）

| 枚举 | 输出格式 | 示例（`0.778-0.258i V`） |
|---|---|---|
| `kRealImaginary` | 实部 + 虚部，**带单位** | `0.778-0.258i V` |
| `kMagDegrees` | 幅度 / 相角（度） | `819.663/-18.3465` |
| `kDbDegrees` | dB 幅度 / 相角（度） | `-1.72729/-18.3465` |
| `kMagRadians` | 幅度 / 相角（弧度） | `819.663/-0.320207` |
| `kDbRadians` | dB 幅度 / 相角（弧度） | `-1.72729/-0.320207` |

说明：
- **只有 `kRealImaginary` 携带单位**（因此也才有缩放）。其余四种是纯 `a/b`
  配对 —— 幅度（或 dB）与相角 —— **不带单位、不做缩放**，因为幅度/相角对
  并不是"Measurement 单位下的一个值"，而实部/虚部是。
- `kRealImaginary` 的实部/虚部**共用一个缩放系数**，且从 `|z|` 解析
  （而非实部 —— 实部接近 0 时会选错前缀）。
- dB 定义为 `20*log10(|z|)`；`|z| == 0` 时输出 `-999`。

### 3.1 复数 × NumberFormat

`number_format` **同样作用于复数**（实部/虚部各自按该模式渲染）：

| NumberFormat | `2.4e9+1e9i Hz` |
|---|---|
| Full | `2400000000+1000000000i Hz` |
| Scientific | `2.4e9+1e9i Hz` |
| Engineering | `2.4+1i GHz` |
| Hex | `0x8f0d1800+0x3b9aca00i` |
| Octal | `021703214000+07346545000i` |
| Binary | `0b10001111000011010001100000000000+0b1110111001101011001010000000000i` |

---

## 4. 其他数据类型

| 类型 | 输出 |
|---|---|
| String | `hello` |
| Boolean | `TRUE` / `FALSE` |
| Vector | `[1,2,3] V`（单位后缀在括号外，只加一次） |
| Matrix | `[[1,2],[3,4]] V` |

向量/矩阵内部元素用 `FormatElement`（不含单位后缀），单位在末尾追加一次。

---

## 5. 默认行为（不传 options）

`to_string()` 无参调用使用 `FormatDefaults::Instance().Get()`，默认为
`{kFull, kRealImaginary, 6, true}`，即**不做任何自动缩放**：

```
0.002 V        -> 0.002 V
1500 V         -> 1500 V
1e9 Hz         -> 1000000000 Hz
42 (int)       -> 42
0.002 (无量纲) -> 0.002
0.778-0.258i V -> 0.778-0.258i V
```

**这是新的回归基线**。历史基线（`kDefault` 自动缩放：`0.002 V -> 2 mV`）已随
`kDefault` 一起移除；依赖该行为的测试改用 `MeasurementFormatTest` 中的
`engineering()` 辅助函数显式切换。

---

## 6. 实现要点

### 6.1 渲染是纯函数

`(value, unit, options) -> string`。**没有 formatter 对象、没有 visitor、没有 per-call 可变状态**，
因此下列函数全部可重入、线程安全。选项始终按 `const&` 传递（16 字节，从不拷贝）。

三种传选项的方式，代价相同：

```cpp
// 1. 全局默认 —— 设一次，之后 to_string() 都用它
FormatDefaults::Instance().Set(o);
m.to_string();

// 2. 一次性 —— 显式传入
m.to_string(o);

// 3. 作用域覆盖 —— RAII，退出（含异常）时恢复，可嵌套
{ FormatScope scope(o);  render_whole_table();  }
```

### 6.2 分层与热点隔离

| 层 | 函数 | 职责 |
|---|---|---|
| 数字 | `FormatElement(double/int, options)` | 纯数字渲染，**不认识单位** |
| 缩放解析 | `ResolveScale(v, unit, options)` | 唯一的昂贵步骤，产出 `DisplayScale` |
| 缩放渲染 | `FormatWithScale(double/int, scale, format, digits)` | 廉价：只套用已解析的 scale + 后缀 |
| 整值 | `Format(m, options)` | 按 `data_kind()` / `data_type()` 分派 |

`DisplayScale` 的 `suffix` 是**预格式化**的：带单位的模式已含前导空格（`" GHz"`），
SPICE 前缀模式**没有**空格（`"G"`），调用方直接拼接即可。

渲染同一单位的大量值（如表格的一整列）时，可以把最贵的一步提到循环外：

```cpp
const DisplayScale s = ResolveScale(magnitude, unit, opts);   // 一次
for (auto& v : column)
    out.push_back(FormatWithScale(v, s, opts.number_format, opts.significant_digits));
```

### 6.3 其他

- 实数（`FormatValue`）与复数（`FormatComplex`）共用 `ResolveScale()`，保证两者
  `number_format` / `show_unit` 规则完全一致。
- `UnitScale` 拆出 `prefix` 字段（`name` = prefix+base，`prefix` = `"G"`），
  使 Engineering 能在隐藏单位时只输出前缀。
- 整数走 `AppendFullInt`（`%d`），**永不出现小数点**；实数 `kFull` 走 `AppendFull`
  （`%.<n>f` + 去尾零）。
- `Unit::to_string()` 由 `UnitRegistry` 记忆化；`best_display()` 扫描扁平的
  `scale_prefix_list()` 并选取**最小**的合格量级。

## 7. 测试

`3rd/xdataset/tests/measurement_test.cc`：

| 用例组 | 数量 | 覆盖 |
|---|---|---|
| `MeasurementFormatTest` | 7 | Engineering 自动选前缀（回归基线） |
| `MeasurementFormatOptionsTest` | 19 | 全部 NumberFormat / ComplexFormat / digits / show_unit |
| `FormatDefaultsTest` | 5 | `FormatDefaults` / `FormatScope`（含嵌套与异常安全） |
| `FormatScaleTest` | 5 | `ResolveScale` + `FormatWithScale` 与直接渲染一致 |

通过 `tests/CMakeLists.txt` 的 `xdataset_test` 目标接入 CTest（LABELS `XDataset`）。
