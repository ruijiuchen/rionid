# RionID+ (Ringed IONs IDentification)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8169341.svg)](https://doi.org/10.5281/zenodo.8169341)

**RionID+** 是一个用于储存环 Schottky 频谱分析的开源软件。它从实验频谱中识别粒子种类，通过模拟粒子在环中的回旋频率，将实验峰与来自 LISE 文件或 AME 数据库的已知核素进行匹配，从而确定离子的质量电荷比（m/q）和产额。

> Forked from [DFreireF/rionid](https://github.com/DFreireF/rionid) with extensive enhancements for GUI-based interactive analysis.

---

## 目录

- [程序结构](#程序结构)
- [架构概览 (MVC)](#架构概览-mvc)
- [模块说明](#模块说明)
- [参数说明](#参数说明)
- [工作流程](#工作流程)
- [运行示例](#运行示例)
- [支持的文件格式](#支持的文件格式)
- [依赖](#依赖)

---

## 程序结构

```
rionid/
├── setup.py                  # 安装脚本
├── setup.cfg                 # 包元数据
├── pyproject.toml            # 构建配置
├── parameters_cache.toml     # 参数缓存文件（自动保存/恢复）
├── README.md                 # 本文档
├── rionid/                   # 核心模型层 (Model)
│   ├── __init__.py           # 导出核心类
│   ├── __main__.py           # CLI 入口
│   ├── version.py            # 版本
│   ├── importdata.py         # 数据导入、模拟、匹配 (核心)
│   ├── nonparams_est.py      # 非参数基线估计 (SNIP, BrPLS)
│   ├── inouttools.py         # 文件 I/O 工具
│   ├── pypeaks.py            # ROOT 峰拟合工具
│   ├── pyqtgraphgui.py       # pyqtgraph 可视化 (View)
│   └── creategui.py          # 旧版 ROOT GUI
└── rionidgui/                # 图形界面层 (View + Controller)
    ├── __init__.py
    ├── __main__.py            # GUI 入口
    ├── gui.py                 # 主窗口 (MainWindow, QSplitter)
    ├── gui_controller.py      # 控制器 (import_controller, 输出保存)
    ├── parameter_gui.py       # 参数面板 (RionID_GUI)
    └── afc_calculator.py      # AFC&gtr 频谱分析面板
```

---

## 架构概览 (MVC)

程序采用经典的 MVC（Model-View-Controller）架构：

```
┌─────────────────────────────────────────────────────────┐
│                    MainWindow (gui.py)                   │
│  ┌─────────────────────┐  ┌───────────────────────────┐  │
│  │  Left: RionID_GUI   │  │  Right: CreatePyGUI       │  │
│  │  (parameter_gui.py) │  │  (pyqtgraphgui.py)        │  │
│  │                     │  │                           │  │
│  │  ┌───────────────┐  │  │  ┌─────────────────────┐  │  │
│  │  │ Mode selector │  │  │  │  实验数据曲线 (蓝)   │  │  │
│  │  │ 参数输入框    │  │  │  │  模拟峰竖线 (彩色)   │  │  │
│  │  │ Load Data     │  │  │  │  阈值线 / 基线      │  │  │
│  │  │ Find Peaks    │  │  │  │  峰标记 (红三角)    │  │  │
│  │  │ Run / SMS/    │  │  │  │  图例 / 光标位置    │  │  │
│  │  │ IMS           │  │  │  │  绘图进度条         │  │  │
│  │  └───────────────┘  │  │  └─────────────────────┘  │  │
│  │        Controller   │  │         View              │  │
│  └─────────┬───────────┘  └───────────────────────────┘  │
└────────────┼──────────────────────────────────────────────┘
             │ visualization_signal / overlay_sim_signal
             ▼
┌─────────────────────────────────────────────────────────┐
│              import_controller (gui_controller.py)        │
│                          Controller                       │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                   ImportData (importdata.py)              │
│                          Model                            │
│  ┌───────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │ 实验数据  │  │ 模拟核素  │  │ 峰检测 & 匹配        │  │
│  │ 导入      │  │ 频率计算  │  │ compute_matches()     │  │
│  │ (NPZ/CSV/ │  │ (srrf)   │  │ scan_match()          │  │
│  │  TDMS/    │  │          │  │ scan_match_brho()     │  │
│  │  ROOT/XML/│  │          │  │ (轻量扫描)            │  │
│  │  TXT)     │  │          │  │                       │  │
│  └───────────┘  └──────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 模块说明

### `rionid/importdata.py` — 核心模型

| 方法 | 功能 |
|------|------|
| `__init__` | 初始化参数，加载实验数据 |
| `_parse_ion_name("205Tl81+")` | 解析离子名 → (质量数, 元素, 电荷) |
| `_get_experimental_data(filename)` | 根据扩展名自动识别格式并导入数据 |
| `load_peaks_summary(filepath)` | **载入峰汇总文件(.txt)** → 构建柱状图存入 experimental_data |
| `detect_peaks_and_widths()` | 峰检测（基于阈值 profile 或固定比例） |
| `_set_particles_to_simulate_from_file(filep)` | 读取 LISE 格式粒子文件 |
| `_calculate_moqs()` | 计算每个离子的 m/q 比值 |
| `_calculate_srrf(fref/brho/ke)` | 计算相对回旋频率 srrf |
| `_simulated_data(harmonics, mode)` | 模拟各谐波的测量频率和产额 |
| `compute_matches(threshold, ...)` | 将实验峰与模拟频率匹配，返回 χ² 和 match_count |
| `scan_match(f_ref, alphap, ...)` | **轻量扫描**：复用粒子缓存，只重算 srrf 和频率 |
| `scan_match_brho(brho, circ, ...)` | **Bρ 轻量扫描**：独立计算每离子回旋频率 |
| `save_matched_result(csv_path)` | 保存详细匹配结果到 CSV |
| `reference_frequency(fref/brho/ke/gam)` | 计算参考粒子的回旋频率 |
| `calc_ref_rev_frequency(...)` | 静态方法：根据 Bρ/KE/γ 计算回旋频率 |
| `_load_threshold_profile(path)` | 加载频率相关阈值 profile |
| `update_threshold_profile_from_clicks(freq, val)` | 从交互点击更新阈值点 |
| `calculate_brho_relativistic(...)` | 从 m/q + 频率计算相对论 Bρ |

**核心理论公式：**

在 Frequency 模式下，相对回旋频率（srrf）为：

```
srrf[i] = 1 - αp × (moq[i] - moq[ref]) / moq[ref]
```

其中 αp 为动量压缩因子（或 γₜ⁻²）。各粒子的测量频率为：

```
f[i] = srrf[i] × f_ref
```

当指定参考谐波数 `ref_harmonic` 时，先算基频 `f₀ = f_ref / h_ref`，再乘当前谐波：

```
f[i] = srrf[i] × f₀ × h
```

在 **Bρ 模式**下，每个离子独立计算其回旋频率，不依赖 srrf 近似：

```
γ = sqrt((Bρ × q × c / (m × 1e6))² + 1)          # 相对论 γ
v = c × sqrt(γ² - 1) / γ                           # 离子速度
f_rev[i] = v / C                                   # 回旋频率 (谐波=1)
f[i] = f_rev[i] × h                                # 第 h 次谐波频率
```

其中：
- `Bρ` — 磁刚度 (Tm)
- `q` — 离子电荷态
- `m` — 离子总质量 (MeV/c²)
- `C` — 储存环周长 (m)
- `h` — 谐波数

χ² 匹配使用产额加权：

```
χ² = Σ(weight_i × Δf_i²) / Σ(weight_i)
```

### `rionid/pyqtgraphgui.py` — 可视化 (View)

| 组件 | 功能 |
|------|------|
| `CreatePyGUI` | 主绘图窗口 (QMainWindow) |
| `plot_experimental_data(data)` | 绘制蓝色实验谱线，可选基线和基线去除曲线 |
| `plot_simulated_data(data)` | 绘制各谐波的离子峰竖线和标签，显示每谐波匹配数 |
| `progressUpdate` 信号 | 绘图进度条信号（显示在 Data File Path 下方） |
| `red_triangles` | 检测到的峰标记（红色三角） |
| `threshold_profile_line/points` | 阈值 profile 曲线和编辑点 |
| `toggle_simulated_data()` | 切换模拟数据显示 |
| `toggle_annotations()` | 切换注释（红/绿/黄标记） |
| `toggle_height_source()` | 切换峰高度来源（实验/模拟） |
| `save_plot_with_dialog()` | 保存为 PNG / PDF / SVG |

### `rionid/nonparams_est.py` — 基线估计

| 方法 | 功能 |
|------|------|
| `SNIP` | 统计敏感非线性迭代峰值去除算法 |
| `BrPLS` | 基线去除的惩罚最小二乘算法（参数 l, ratio） |

### `rionidgui/parameter_gui.py` — 参数面板 (Controller)

| 组件 | 功能 |
|------|------|
| `RionID_GUI` | 左侧参数面板 (QWidget) |
| `📂 Load Data` | **载入实验数据**：.txt 文件弹出柱状图配置对话框，设置 x 范围/bin 宽度/预览 |
| `🔍 Find Peaks` | **在已载入数据上执行峰检测** |
| `Apply Baseline Once` | 应用基线去除一次并缓存 |
| `Threshold Profile` | 阈值 profile 路径 / Browse / Start Click Threshold / Clear |
| `run_script()` | "Run" 按钮：执行单次模拟 + 匹配，显示每谐波匹配数 |
| `SMS_pid_script()` | "Run SMS"：自动扫描 f_ref 和 alphap（αₚ），带进度条 |
| `IMS_pid_script()` | "Run IMS"：自动扫描 Bρ 和 ring circumference，带进度条 |
| `HistogramConfigDialog` | 柱状图参数设置对话框（x 范围、bin 宽度、预览） |
| `enterPlotPickMode()` | 进入图谱点击取频率模式 |

### `rionidgui/afc_calculator.py` — AFC&gtr Calculator

AFC 频谱分析面板，提供实验时频谱数据的可视化与谐振参数提取。

| 功能按钮 | 说明 |
|---------|------|
| **Load Data & Plot** | 载入实验时频谱（.npz/.root）和电压事件文件，绘制 2D 时频谱 |
| **Project** | 按电压事件切分时频谱，提取每个事件的频率投影 |
| **Find Peaks** | 在每个投影上检测峰，计算 **每帧面积误差** `area_err`（帧间波动标准差 × √N），输出 `afc_peaks.csv` |
| **Fit Har** | 谐波数线性拟合，输出 `afc_harmonics.csv`（含 `area_err`） |
| **Plot Norm Area** | **面积比谐振拟合**：以选定参考峰（1st / last / Max Area）为分母构建频对 (f₁, f₂)，洛伦兹比值模型 `R = \|H(f₁)\|²/\|H(f₂)\|²`，加权 `curve_fit` + **500 次蒙特卡洛** 模拟，输出 Q 与 f_sys 及其不确定度。**主界面可设 Q0/f0 初始参数，对话框内可 Re-fit。** |
| **Area vs Time** | 选定谐波的面积-时间曲线（含误差棒），使用 H(f) 模型做 AFC 修正（不归一化），加权 A0*exp(-λt) 拟合。自动从 afc_harmonics.csv 读取数据。 |
| **Self-consistent** | 与 Plot Norm Area 相同的比值拟合法迭代 + 加权指数衰减拟合。自动从 afc_harmonics.csv 读取谐波数据。 |

**数据流**：`afc_peaks.csv` ← (Find Peaks) ← 频谱数据 → (Fit Har) → `afc_harmonics.csv`

### `rionidgui/gui_controller.py` — 控制器

| 函数 | 功能 |
|------|------|
| `import_controller(...)` | 串联所有计算步骤，返回 ImportData 对象 |
| `display_nions(nions, ...)` | 按产额排序截断显示指定数量的离子 |
| `save_simulation_results(...)` | 保存模拟结果到文件 |

---

## 参数说明

### 文件选择区

| 参数 | 说明 |
|------|------|
| **Experimental Data File** | 实验频谱数据文件（.npz/.csv/.tdms/.root/.xml/.txt） |
| **.lpp File** | LISE 格式粒子列表文件（含各核素产额） |

### 数据处理选项

| 参数 | 说明 |
|------|------|
| **Remove baseline** | 启用基线去除（使用 BrPLS 算法） |
| **l(e.g. 1000000)** | BrPLS 基线去除参数 λ（平滑度） |
| **ratio(e.g. 1e-6)** | BrPLS 基线去除参数 ratio（非对称权重） |

### 模式选择

| 参数 | 说明 |
|------|------|
| **Mode** | 计算模式：Frequency / Bρ / Kinetic Energy |
| **Value** | 对应模式的参考值（频率 Hz / 磁刚度 Tm / 动能 MeV/u） |

### 物理参数

| 参数 | 说明 |
|------|------|
| **αp or γt** | 动量压缩因子 αp（若输入 >1 则自动转换为 1/γₜ²） |
| **Circumference (m)** | 储存环周长（ESR 约 108.36 m） |
| **Scaling factor** | 模拟产额的缩放因子 |

### 离子参数

| 参数 | 说明 |
|------|------|
| **Ref. Harmonic** | 参考频率对应的谐波数（如 125），用于计算基频 f₀ |
| **Fundamental f0 (Hz)** | 自动计算的基频（谐波=1 的频率），只读 |
| **Reference ion (AAEl+QQ)** | 参考离子，格式如 `205Tl81+`、`86Kr25+` |
| **Harmonics (e.g. 124 125 126)** | 要模拟的谐波列表（空格分隔） |
| **Harmonic Frequencies (Hz)** | 自动计算的各谐波频率，只读 |
| **Highlight ions** | 高亮显示的离子（逗号分隔） |
| **Number of ions to display** | 显示产额最高的 N 个离子 |

### 峰检测参数

| 参数 | 说明 |
|------|------|
| **Peak threshold (of max)** | 峰检测阈值（最大幅度的百分比，如 0.05） |
| **Peak min distance (Hz)** | 峰最小距离（Hz），防止检测到假峰 |
| **Peak search range min/max (Hz)** | 峰搜索频率范围（支持从图谱 Pick） |

### 匹配参数

| 参数 | 说明 |
|------|------|
| **Sim. - Exp. max. distance (Hz)** | 匹配阈值：模拟频率与实验频率的最大允许偏差 |
| **Second-order correction** | 二阶校正系数（a0 a1 a2） |

### SMS 设置

| 参数 | 说明 |
|------|------|
| **αp or γt min / max / step** | αp 扫描范围与步长 |
| **Reference frequency min/max (Hz)** | 参考频率扫描范围（支持从图谱 Pick） |

### IMS 设置

| 参数 | 说明 |
|------|------|
| **Bρ min / max / step** | 磁刚度扫描范围与步长 |
| **Circumference min / max / step** | 周长扫描范围与步长 |

### 输出文件

| 参数 | 说明 |
|------|------|
| **Simulation result** | 模拟结果保存路径 |
| **matched result** | 详细匹配结果 CSV 保存路径 |

---

## 工作流程

### 标准流程

```
📂 Load Data  →  从文件载入数据
                 .txt 文件弹出柱状图对话框 → 设置 x 范围/bin 宽度 → 预览 → 确认
                 柱状图参数自动保存到 parameters_cache.toml，下次自动恢复

🔍 Find Peaks  →  在柱状图上检测峰 → 标注红色三角标记
                 （Load Data 不自劋寻峰）

Run            →  用已检测的峰进行模拟匹配
                 输出各谐波匹配数、总 χ²、总匹配数
                 Legend 显示每谐波匹配数

Run SMS        →  自动扫描 f_ref × alphap 的最佳组合
                 显示进度条，找到最大匹配数

Run IMS        →  自动扫描 Bρ × circumference 的最佳组合
                 显示进度条，找到最大匹配数
```

### 峰汇总文件 (.txt)

RionID+ 支持直接载入 `all_peaks_summary.txt` 格式的峰汇总文件（来自频谱分析流水线的输出）。

文件格式（制表符分隔）：
```
总序号    峰序号    频率[MHz]    峰高    FWHM[MHz]    start[s]    ...
1         1         309.369946   0.000536  9.536743e-05  0.000000   ...
```

程序将第 3 列频率分布构建为柱状图（直方图），存入 `experimental_data`。

### Threshold Profile 交互编辑

1. 在 **Threshold Profile** 行点击 **Browse** 选择 `height_thresh.csv`
2. 点击 **Start Click Threshold** 按钮进入点击模式
3. 在频谱图上 **左键点击** 添加/更新阈值点（绿色星形）
4. **右键**已有阈值点删除
5. 点击 **Clear** 清除所有阈值点

---

## 运行示例

### 1. 启动 GUI

```bash
python -m rionidgui
```
或直接运行：
```bash
rionidgui
```

### 2. 基本使用流程

1. **📂 Load Data** — 选择实验数据文件 → 自动载入并绘图
2. **🔍 Find Peaks** — 在数据上检测峰
3. 设置物理参数（模式、参考离子、谐波、αp 等）
4. 点击 **Run** 执行模拟与匹配

### 3. SMS mode 自动扫描

自动扫描参考频率和 αp 的最佳组合：

1. 设置 SMS 扫描范围（αp min/max/step, fref min/max）
2. 点击 **Run SMS**（显示进度条）
3. 程序自动遍历所有组合，按 match_count 降序 / χ² 升序选择最佳参数

### 4. IMS mode 自动扫描

自动扫描 Bρ 和环周长的最佳组合：

1. 设置 IMS 扫描范围（Bρ min/max/step, Circ min/max/step）
2. 点击 **Run IMS**（显示进度条）
3. 程序遍历所有组合，显示最佳匹配结果

---

## 支持的文件格式

### 实验数据

| 扩展名 | 格式说明 |
|--------|---------|
| `.npz` | NumPy 压缩格式（TIQ 或 spectrum 类型） |
| `.csv` | 逗号分隔的频谱数据 |
| `.tdms` | NI TDMS 二进制格式 |
| `.bin_fre` / `.bin_amp` / `.bin_time` | TDMS 二进制文件 |
| `.root` | ROOT 格式（含 2D 直方图） |
| `.xml` / `.Specan` | RSA 频谱分析仪 XML 格式 |
| `.txt` | **峰汇总文件**（列格式频率/MHz 峰高 FWHM）→ 构建柱状图 |

### 粒子列表

| 扩展名 | 格式说明 |
|--------|---------|
| `.lpp` | LISE 格式粒子列表 |

---

## 依赖

- [Barion](https://github.com/xaratustrah/barion) — 基本粒子物理计算
- [LISEreader](https://github.com/gwgwhc/lisereader) — LISE 文件解析
- [PyQt5](https://pypi.org/project/PyQt5/) — GUI 框架
- [pyqtgraph](http://pyqtgraph.org/) — 科学绘图
- [NumPy / SciPy](https://scipy.org/) — 数值计算
- [loguru](https://github.com/Delgan/loguru) — 日志系统
- [toml](https://pypi.org/project/toml/) — 配置参数格式
- iqtools — RSA 数据读取

---

## 版本

当前版本：**5.3.0**（新增柱状图支持、Load Data / Find Peaks 分离工作流、IMS 模式、绘图进度条等）

## 致谢

- Dr. RuiJiu Chen ([@chenruijiu](https://github.com/chenruijiu/)) — 原始 C++ ToF 模拟代码
- Dr. Shahab Sanajri ([@xaratustrah](https://github.com/xaratustrah/)) — 软件架构指导
- G. Hudson-Chang ([@gwgwhc](https://github.com/gwgwhc/)) — LISEreader 和教程
