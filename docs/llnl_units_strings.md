# llnl-units 支持的单位字符串

**最后修改：** 2026/07/17

本文档列出 llnl-units（v0.13.1）中 `unit_from_string()` 可解析的所有单位字符串，以及 `to_string()` 可还原的命名单位。

数据来源于 `units_conversion_maps.hpp` 中的三个核心表：
- `defined_unit_names_si`（113 条）—— SI 及常用导出单位的 `to_string` 显示名
- `defined_unit_names_customary`（55 条）—— 美制/英制单位的 `to_string` 显示名
- `defined_unit_strings_si`（1213 条）—— `unit_from_string` 可解析的 SI/通用单位
- `defined_unit_strings_customary`（1181 条）—— `unit_from_string` 可解析的美制/英制单位

**总计约 2500+ 个可解析字符串**（含别名、复数形式、大小写变体、unicode 符号）。

---

## 记号约定

| 记号 | 含义 |
|---|---|
| `[仿射]` | 带偏移的单位（degC/degF），不能简单用 `value * multiplier` 转换 |
| `×1000` | multiplier ≠ 1（非相干 SI），如 km 的 multiplier = 1000 |
| `(…^n)` | 导出单位的幂次 |

---

## 1. 无量纲 / 计数

| 字符串 | 等价单位 |
|---|---|
| `""`, `"1"`, `"one"`, `"[]"`, `"def"`, `"defunit"` | 无量纲（1） |
| `"%"`, `"percent"`, `"pct"`, `"%age"` | %（0.01） |
| `"ppm"`, `"PPM"`, `"[PPM]"`, `"partspermillion"` | ppm（1e-6） |
| `"ppb"`, `"PPB"`, `"partsperbillion"` | ppb（1e-9） |
| `"ppt"`, `"pptr"`, `"partspertrillion"` | 万亿分之一（1e-12） |
| `"ppq"`, `"partsperquadrillion"` | 千万亿分之一（1e-15） |
| `"pu"`, `"perunit"` | per-unit（1） |
| `"count"`, `"unit"`, `"#`, `"number"`, `"item"`, `"piece"`, `"part"` | 计数 |
| `"ratio"`, `"rat"` | 比值 |
| `"pair"`, `"dozen"`, `"score"`, `"gross"` | 2, 12, 20, 144 |
| `"half"`, `"quarter"`, `"third"`, `"fourth"`, `"tenth"` | 分数 |
| `"permille"`, `"‰"` | 千分之一 |
| `"bp"`, `"basispoint"` | 万分之一 |
| `"inf"`, `"infinity"`, `"∞"` | 无穷大 |
| `"nan"`, `"NaN"` | NaN |
| `"pi"`, `"PI"`, `"π"` | 圆周率 |

---

## 2. SI 前缀

前缀可与任意单位组合（如 `km`、`GHz`、`mL`、`kV`），llnl-units 自动拆分识别。

| 简写 | 量级 | 简写 | 量级 |
|---|---|---|---|
| `Y` | 1e24 | `y` | 1e-24 |
| `Z` | 1e21 | `z` | 1e-21 |
| `E` | 1e18 | `a` | 1e-18 |
| `P` | 1e15 | `f` | 1e-15 |
| `T` | 1e12 | `p` | 1e-12 |
| `G` | 1e9  | `n` | 1e-9 |
| `M` | 1e6  | `µ` / `u` | 1e-6 |
| `k` | 1e3  | `m` | 1e-3 |
| `h` | 1e2  | `c` | 1e-2 |
| `da` | 1e1  | `d` | 1e-1 |

---

## 3. SI 基本单位

| 量 | 字符串 |
|---|---|
| 长度 | `"m"`, `"meter"`, `"metre"` |
| 质量 | `"kg"`, `"KG"`, `"kilogram"` |
| 时间 | `"s"`, `"sec"`, `"second"` |
| 电流 | `"A"`, `"amp"`, `"amps"`, `"ampere"` |
| 温度 | `"K"`, `"kelvin"`, `"kelvins"`, `"degK"` 等 |
| 物质的量 | `"mol"`, `"mole"` |
| 发光强度 | `"cd"`, `"candela"` |

---

## 4. 长度 / 距离

### SI 及常用

| 字符串 | 说明 |
|---|---|
| `"m"`, `"meter"`, `"metre"` | 米 |
| `"km"`, `"kilometer"` | 千米（×1000） |
| `"cm"`, `"centimeter"` | 厘米 |
| `"mm"`, `"millimeter"` | 毫米 |
| `"nm"`, `"nanometer"` | 纳米 |
| `"dm"` | 分米 |
| `"pm"` | 皮米 |
| `"fm"` | 飞米 |
| `"micron"` | 微米 |
| `"fermi"` | 飞米 |
| `"Å"`, `"angstrom"` | 埃（1e-10 m） |

### 天文

| 字符串 | 说明 |
|---|---|
| `"ly"`, `"lightyear"`, `"light-year"` | 光年 |
| `"au"`, `"AU"`, `"astronomicalunit"` | 天文单位 |
| `"pc"`, `"parsec"` | 秒差距 |
| `"[LY]"`, `"[c]"`, `"c"` | 光速 / 光年常量 |

---

## 5. 面积

| 字符串 | 说明 |
|---|---|
| `"m^2"`, `"m2"`, `"sq m"` | 平方米 |
| `"km^2"`, `"cm^2"`, `"mm^2"` | 平方千米/厘米/毫米 |
| `"are"`, `"ar"`, `"AR"` | 公亩（100 m²） |
| `"hectare"`, `"ha"` | 公顷（10000 m²） |
| `"barn"`, `"b"`, `"BRN"` | 靶恩（1e-28 m²） |
| `"acre"` | 英亩 |

---

## 6. 体积 / 容积

| 字符串 | 说明 |
|---|---|
| `"m^3"`, `"m3"` | 立方米 |
| `"cm^3"`, `"cc"` | 立方厘米 |
| `"L"`, `"l"`, `"liter"`, `"litre"` | 升 |
| `"mL"`, `"ml"` | 毫升 |
| `"dL"` | 分升 |
| `"uL"` | 微升 |
| `"st"`, `"stere"` | 立方 meter（木材） |

---

## 7. 时间

| 字符串 | 说明 |
|---|---|
| `"s"`, `"sec"`, `"second"` | 秒 |
| `"ms"`, `"millisecond"` | 毫秒 |
| `"min"`, `"minute"` | 分钟 |
| `"h"`, `"hr"`, `"hour"` | 小时 |
| `"day"`, `"d"`, `"dy"`, `"D"` | 天 |
| `"week"`, `"wk"`, `"WK"` | 周 |
| `"yr"`, `"y"`, `"year"` | 年（365 天） |
| `"a_j"`, `"year_j"` | 儒略年（365.25 天） |
| `"a_g"`, `"year_g"` | 格里高利年（365.2425 天） |
| `"a_t"`, `"year_t"` | 回归年 |
| `"syr"`, `"year_sdr"` | 恒星年 |
| `"month"`, `"mo"` | 月（格里高利，1/12 年） |
| `"month_[30]"` | 30 天 |
| `"leapyear"` | 闰年（366 天） |
| `"decade"` | 十年 |
| `"century"` | 百年 |
| `"millennium"` | 千年 |
| `"eon"` | 十亿年 |
| `"workyear"`, `"workmonth"`, `"workweek"`, `"workday"` | 工作年/月/周/日 |
| `"shake"` | 10 ns |
| `"jiffy"` | 0.01 s |

---

## 8. 频率

| 字符串 | 说明 |
|---|---|
| `"Hz"`, `"hertz"` | 赫兹 |
| `"kHz"`, `"MHz"`, `"GHz"` | 千/兆/吉赫兹 |
| `"rpm"` | 转/分钟 |
| `"rps"` | 转/秒 |
| `"rev"`, `"revs"`, `"revolution"` | 转数 |
| `"r"` | 转 |

---

## 9. 速度

| 字符串 | 说明 |
|---|---|
| `"m/s"`, `"mps"` | 米/秒 |
| `"km/h"`, `"kph"` | 千米/时 |
| `"mph"` | 英里/时 |
| `"knot"`, `"kn"` | 节（海里/时） |
| `"knot_br"`, `"kn_br"` | 英制节 |

---

## 10. 加速度

| 字符串 | 说明 |
|---|---|
| `"m/s^2"` | 米/秒² |
| `"gravity"`, `"gp"`, `"freefall"` | 标准重力加速度（9.80665 m/s²） |

---

## 11. 质量

| 字符串 | 说明 |
|---|---|
| `"kg"`, `"kilogram"` | 千克 |
| `"g"`, `"gm"`, `"gram"` | 克 |
| `"mg"`, `"milligram"` | 毫克 |
| `"mcg"` | 微克 |
| `"gamma"` | 微克 |
| `"t"`, `"tonne"`, `"ton_m"`, `"mt"` | 公吨（1000 kg） |
| `"u"`, `"amu"`, `"Da"`, `"dalton"` | 原子质量单位 |
| `"carat"`, `"ct"`, `"karat"` | 克拉（0.2 g） |
| `"carat_m"`, `"ct_m"` | 公制克拉 |
| `"lb"`, `"lbs"`, `"pound"` | 磅 |
| `"oz"`, `"ounce"` | 盎司 |
| `"ton"` | 短吨（2000 lb） |
| `"grain"` | 格令 |
| `"slug"` | 斯勒格 |
| `"stone"` | 英石 |

---

## 12. 力

| 字符串 | 说明 |
|---|---|
| `"N"`, `"newton"` | 牛顿 |
| `"lbf"` | 磅力 |
| `"dyne"`, `"dyn"` | 达因（1e-5 N） |
| `"pond"` | 克力 |
| `"kp"` | 千克力 |
| `"gf"` | 克力 |

---

## 13. 压强

| 字符串 | 说明 |
|---|---|
| `"Pa"`, `"pa"`, `"pascal"` | 帕斯卡 |
| `"kPa"`, `"MPa"`, `"GPa"` | 千/兆/吉帕 |
| `"bar"`, `"BAR"` | 巴（1e5 Pa） |
| `"mbar"` | 毫巴 |
| `"atm"`, `"atmosphere"`, `"standardatmosphere"` | 标准大气压 |
| `"psi"` | 磅/平方英寸 |
| `"psig"` | 表压（psi 相对大气压） |
| `"psia"` | 绝对压（psi） |
| `"Pa_g"`, `"Pa_a"` | 表压/绝对压（Pa） |
| `"torr"`, `"Torr"` | 托（mmHg） |
| `"mmHg"`, `"mm_Hg"` | 毫米汞柱 |
| `"inHg"` | 英寸汞柱 |
| `"mmH2O"`, `"mm_H2O"` | 毫米水柱 |
| `"inH2O"` | 英寸水柱 |
| `"ksi"` | 千磅/平方英寸 |
| `"barye"`, `"Ba"` | 微巴（CGS） |

---

## 14. 能量 / 功 / 热量

| 字符串 | 说明 |
|---|---|
| `"J"`, `"joule"`, `"Joule"` | 焦耳 |
| `"kJ"`, `"MJ"`, `"GJ"` | 千/兆/吉焦 |
| `"eV"`, `"electronvolt"` | 电子伏特 |
| `"keV"`, `"MeV"`, `"GeV"` | 带电伏 |
| `"hartree"`, `"Eh"`, `"E_h"`, `"Ha"` | Hartree 能量 |
| `"Ry"` | Rydberg（13.60583 eV） |
| `"cal"` | 卡路里 |
| `"kcal"` | 千卡 |
| `"btu"` | 英热单位 |
| `"therm"` | 瑟姆 |
| `"tonc"` | 吨 TNT 当量 |
| `"erg"`, `"ERG"` | 尔格（1e-7 J） |
| `"Wh"` | 瓦时 |
| `"kWh"`, `"kwh"` | 千瓦时 |
| `"MWh"` | 兆瓦时 |
| `"foeb"` | 油当量桶 |

---

## 15. 功率

| 字符串 | 说明 |
|---|---|
| `"W"`, `"watt"` | 瓦特 |
| `"kW"`, `"kilowatt"` | 千瓦 |
| `"MW"`, `"megawatt"` | 兆瓦 |
| `"mW"`, `"milliwatt"` | 毫瓦 |
| `"GW"` | 吉瓦 |
| `"hp"`, `"horsepower"` | 马力 |
| `"EER"` | 能效比 |
| `"ton_refrig"` | 冷吨 |
| `"VA"`, `"VA"` | 伏安 |

---

## 16. 电学

| 量 | 字符串 |
|---|---|
| 电压 | `"V"`, `"volt"`, `"kV"`, `"kilovolt"`, `"mV"` |
| 电流 | `"A"`, `"amp"`, `"amps"`, `"mA"`, `"milliamp"` |
| 电阻 | `"ohm"`, `"Ohm"`, `"Ω"`, `"kilohm"`, `"megohm"` |
| 电导 | `"S"`, `"siemens"`, `"mho"` |
| 电容 | `"F"`, `"farad"` |
| 电感 | `"H"`, `"henry"`, `"henries"` |
| 电荷 | `"C"`, `"coulomb"` |
| 磁通量 | `"Wb"`, `"weber"` |
| 磁感应强度 | `"T"`, `"tesla"` |
| 电量 | `"Ah"`, `"Ahr"`, `"Wh"`, `"kWh"` |
| 电抗 | `"VAR"`, `"var"`, `"MVAR"`, `"MVA"` |
| PU 值 | `"puV"`, `"puA"`, `"puW"`, `"puMW"`, `"puOhm"`, `"puHz"` |

---

## 17. 角度 / 立体角

| 字符串 | 说明 |
|---|---|
| `"rad"`, `"radian"` | 弧度 |
| `"sr"`, `"steradian"` | 球面度 |
| `"deg"`, `"°"`, `"o"` | 度 |
| `"arcmin"`, `"arcminute"`, `"MOA"`, `"'"` | 角分 |
| `"arcsec"`, `"arcsecond"`, `"asec"`, `"\""` | 角秒 |
| `"mas"` | 毫角秒 |
| `"uas"` | 微角秒 |
| `"grad"`, `"gon"` | 百分度 |
| `"circ"`, `"circle"`, `"turn"`, `"cycle"`, `"revolution"` | 圈（2π rad） |
| `"quadrant"` | 象限（90°） |
| `"sph"`, `"sphere"` | 球面（4π sr） |

---

## 18. 温度

| 字符串 | 说明 |
|---|---|
| `"K"`, `"kelvin"`, `"degK"`, `"degKelvin"`, `"°K"` | 开尔文 |
| `"degC"`, `"°C"`, `"celsius"`, `"oC"` | 摄氏度 **[仿射]** |
| `"degF"`, `"°F"`, `"fahrenheit"` | 华氏度 **[仿射]** |
| `"degR"`, `"rankine"` | 兰氏度 **[仿射]** |

---

## 19. 光学

| 字符串 | 说明 |
|---|---|
| `"cd"`, `"candela"` | 坎德拉 |
| `"lm"`, `"lumen"` | 流明 |
| `"lx"`, `"lux"`, `"luxes"` | 勒克斯 |
| `"nit"`, `"nt"` | 尼特（cd/m²） |
| `"sb"`, `"stilb"` | 熙提（CGS） |
| `"ph"`, `"phot"` | 辐透（CGS） |
| `"lambert"`, `"Lb"` | 朗伯 |
| `"footcandle"`, `"fc"` | 英尺烛光 |
| `"candle"`, `"candlepower"`, `"CP"` | 坎德拉（旧称） |

---

## 20. 辐射 / 核物理

| 字符串 | 说明 |
|---|---|
| `"Bq"`, `"becquerel"` | 贝克勒尔 |
| `"Ci"`, `"curie"` | 居里（3.7e10 Bq） |
| `"Gy"`, `"gray"` | 戈瑞 |
| `"Sv"`, `"sievert"` | 希沃特 |
| `"R"`, `"roentgen"` | 伦琴（2.58e-4 C/kg） |
| `"RAD"`, `"radiationabsorbeddose"` | 拉德 |
| `"rem"`, `"REM"` | 雷姆 |
| `"rutherford"` | 卢瑟福（1e6 Bq） |
| `"activity"` | 活度 |
| `"barn"`, `"b"` | 靶恩（截面） |

---

## 21. 物质/化学/实验

| 字符串 | 说明 |
|---|---|
| `"mol"`, `"mole"`, `"eq"`, `"Eq"`, `"equivalent"` | 摩尔/当量 |
| `"M"`, `"molar"` | 摩尔浓度（mol/L） |
| `"kat"`, `"katal"` | 卡塔（催化活性） |
| `"pH"`, `"pHscale"` | pH |
| `"U"`, `"units"`, `"enzymeunit"` | 酶活单位 |
| `"IU"`, `"[IU]"`, `"[iU]"`, `"internationalunit"` | 国际单位 |
| `"arb. unit"`, `"arbu"`, `"arbitraryunit"` | 任意单位 |
| `"pfu"`, `"PFU"`, `"plaqueformingunit"` | 蚀斑形成单位 |
| `"TCID50"`, `"CCID50"`, `"EID50"` | 半数感染剂量 |

---

## 22. 数据 / 信息

| 字符串 | 说明 |
|---|---|
| `"bit"`, `"BIT"` | 比特 |
| `"B"`, `"byte"`, `"By"` | 字节 |
| `"kB"`, `"MB"`, `"GB"`, `"TB"` | SI 前缀字节 |
| `"KiB"`, `"MiB"`, `"GiB"` | 二进制前缀字节 |
| `"nibble"`, `"nybble"` | 半字节（4 bit） |
| `"bps"`, `"baud"`, `"Bd"` | 比特/秒 |
| `"Sh"`, `"shannon"` | 香农 |
| `"Hart"`, `"hartley"` | 哈特利 |
| `"dB"`, `"decibel"`, `"DB"` | 分贝 |
| `"Np"`, `"neper"` | 奈培 |
| `"octave"` | 倍频程 |
| `"flop"`, `"flops"`, `"mips"` | 浮点/指令运算 |

---

## 23. 声学 / 分贝参考

|\| 字符串 | 参考 |
|---|---|
| `"dB_SPL"`, `"decibel_SPL"`, `"DBSPL"` | 声压级（20 µPa） |
| `"dB[V]"`, `"decibelV"`, `"dB(V)"` | dBV（1 V） |
| `"dB[mV]"`, `"dB(mV)"` | dBmV（1 mV） |
| `"dB[uV]"`, `"decibelmicrovolt"` | dBµV（1 µV） |
| `"dB[W]"`, `"dB(W)"` | dBW（1 W） |
| `"B[kW]"`, `"dB[kW]"` | dBkW（1 kW） |
| `"dBZ"`, `"BZ"` | dBZ（雷达反射率） |

---

## 24. 物理常数

所有物理常数用 `[]` 包裹表示：

| 字符串 | 常数 |
|---|---|
| `"[c]"`, `"c"`, `"speedoflight"` | 光速 |
| `"[h]"`, `"[hbar]"`, `"hbar"` | 普朗克常数 / 约化普朗克常数 |
| `"[k]"`, `"[K]"` | 玻尔兹曼常数 |
| `"[e]"`, `"elementarycharge"` | 元电荷 |
| `"[G]"` | 引力常数 |
| `"[g]"`, `"[g0]"`, `"standardgravity"` | 标准重力加速度 |
| `"[me]"`, `"m_e"`, `"electronmass"` | 电子质量 |
| `"[mp]"`, `"m_p"`, `"protonmass"` | 质子质量 |
| `"[mn]"`, `"m_n"`, `"neutronmass"` | 中子质量 |
| `"[Na]"` | 阿伏伽德罗常数 |
| `"[R]"` | 气体常数 |
| `"[eps_0]"`, `"vacuumpermittivity"` | 真空介电常数 |
| `"[mu0]"`, `"mu_0"` | 真空磁导率 |
| `"[alpha]"` | 精细结构常数 |
| `"[a0]"` | 玻尔半径 |
| `"[Rinf]"` | 里德伯常数 |
| `"[Kj]"` | 约瑟夫森常数 |
| `"[Rk]"` | 冯·克利青常数 |
| `"[phi0]"` | 磁通量子 |
| `"plancklength"`, `"planckmass"`, `"plancktime"`, `"planckcharge"`, `"plancktemperature"` | 普朗克单位 |

---

## 25. 其他单位分类

### 流量
`"CFM"`（立方英尺/分）、`"LPM"`（升/分）、`"LPS"`（升/秒）、`"sverdrup"`（1e6 m³/s）

### 密度
`"kg/m^3"`

### 运动粘度
`"St"`, `"stoke"`（斯托克斯）、`"cps"`（厘泊）、`"P"`, `"poise"`（泊）

### 剂量/医学
`"[S]"`, `"svedberg"`、`"diopter"`, `"dpt"`、`"HPF"`, `"LPF"`（显微镜视野）、`"drop"`、`"Ch"`, `"french"`（导管规格）、`"mesh"`、`"clo"`（服装热阻）、`"MET"`（代谢当量）、`"[hnsf'U]"`（CT 亨氏单位）、`"woodunit"`（肺血管阻力）、`"pru"`（外周阻力单位）

### 地震
`"moment_magnitude"`（矩震级）

### 石油/天然气
`"bbl"`, `"barrel"`、`"Sm^3"`, `"Nm^3"`（标准/普通立方米）、`"SCF"`, `"MCF"`, `"MMCF"`, `"BCF"`, `"TCF"`（标准立方英尺）、`"degAPI"`、`"degBaume"`、`"GtC"`（十亿吨碳）

### 货币
`"$"`, `"dollar"`, `"dollar_us"`, `"USD"`、`"euro"`、`"yen"`、`"ruble"`、`"M$"`, `"mil$"`（百万美元）、`"B$"`, `"bil$"`（十亿美元）、`"$/MWh"`, `"$/kWh"`

### 气候
`"gwp"`（全球变暖潜势）、`"gtp"`（全球温变潜势）、`"ode"`（臭氧消耗当量）

### 分贝对数
`"Np"`, `"neper"`（奈培）、`"octave"`（倍频程）、`"erlang"`（话务量）

### 其他
- **亮度**：`"asb"`, `"apostilb"`, `"blondel"`, `"bril"`, `"skot"`, `"ftL"`, `"footlambert"`
- **磁学**：`"G"`, `"Gs"`, `"gauss"`（高斯）、`"Mx"`, `"maxwell"`（麦克斯韦）、`"Oe"`, `"oersted"`（奥斯特）、`"gilbert"`, `"Gb"`, `"Gi"`（吉伯）
- **CGS 静电**：`"statC"`, `"Fr"`, `"franklin"`, `"esu"`, `"emu"`
- **CGS 电磁**：`"abamp"`, `"abA"`, `"abvolt"`, `"abV"`, `"abohm"`, `"abF"`, `"statvolt"`, `"stV"`, `"statohm"`, `"statF"`
- **振动**：`"ASD"`（加速度谱密度）、`"rootHertz"`, `"Hz^(1/2)"`（根赫兹）
- **方向**：`"north"`, `"south"`, `"east"`, `"west"`、`"degN"`, `"degS"`, `"degE"`, `"degW"`, `"degT"`, `"true"`
- **糖度**：`"degBrix"`, `"degBalling"`

---

## 26. 复数 / 大小写 / 别名

llnl-units 对每个单位都注册了大量变体，例如：

- `"second"` / `"seconds"`（复数但功能等价）
- `"meter"` / `"metre"`（美式/英式拼写）
- `"in"` / `"inch"` / `"inches"` / `"in_i"` / `"in_US"` / `"IN_US"` / `"IN_I"`
- `"ft"` / `"foot"` / `"feet"` / `"ft_i"` / `"ft_US"` / `"FT_I"` / `"foot_i"`
- `"gal"` / `"gal_US"` / `"gal_br"` / `"GAL_BR"`（US/UK 加仑有不同 multiplier）
- `"yr"` / `"year"` / `"year_j"` / `"year_g"` / `"year_t"` / `"syr"`（不同类型年份）

**通常只记规范名即可，变体自动支持。**

---

## 27. `to_string` 可还原的命名单位（168 条）

`to_string` 对被 canonicalize 后的单位会反查以下映射表还原可读名。此表仅覆盖命名单位——纯前缀组合（如 `kV`）由规则自动生成。

### SI / 通用单位（113 条）

`m`, `m^2`, `m^3`, `kg`, `mol`, `dm`, `dL`, `A`, `Ah`, `V`, `s`, `Bs`, `cd`, `K`, `N`, `Pa`, `J`, `C`, `F`, `(1000MF)`, `(1000YT)`, `(1000000YT)`, `S`, `Wb`, `T`, `H`, `(A^-2*pJ)`, `lm`, `lux`, `R`, `Ci`, `ZL`, `bar`, `min`, `ms`, `h`, `day`, `week`, `yr`, `deg`, `rad`, `°C`, `degC^2`, `cm`, `km`, `km^2`, `mm`, `nm`, `strain`, `ly`, `au`, `%`, `ASD`, `rootHertz`, `$`, `count`, `ratio`, `ERROR`, `defunit`, `flag`, `eflag`, `pu`, `Gy`, `Sv`, `Hz`, `kHz`, `rpm`, `kat`, `sr`, `W`, `VAR`, `MVAR`, `MW`, `kW`, `mW`, `puMW`, `puW`, `puV`, `puA`, `mA`, `kV`, `are`, `hectare`, `barn`, `puOhm`, `puHz`, `eV`, `atm`, `mmHg`, `mmH2O`, `arb. unit`, `[IU]`, `kWh`, `MWh`, `M$`, `B$`, `L`, `mL`, `uL`, `t`, `u`, `kB`, `MB`, `GB`, `KiB`, `MiB`, `Å`, `g`, `mg`, `(eV*[c]^-2)`, `GiB`, `ppm`, `ppb`

### 美制/英制单位（55 条）

`in`, `in^2`, `in^3`, `ft`, `ft_br`, `in_br`, `yd_br`, `rd_br`, `mi_br`, `pc_br`, `lk_br`, `ch_br`, `nmi_br`, `kn_br`, `knot`, `ft^2`, `ft^3`, `°F`, `yd`, `rd`, `yd^2`, `yd^3`, `syr`, `a_g`, `a_t`, `a_j`, `grad`, `mi`, `mi^2`, `acre`, `therm`, `tonc`, `hp`, `mph`, `kcal`, `btu`, `CFM`, `psi`, `psig`, `inHg`, `inH2O`, `torr`, `EER`, `gal`, `bbl`, `lb`, `ton`, `bu`, `floz`, `oz`, `cup`, `tsp`, `tbsp`, `qt`, `lbf`
