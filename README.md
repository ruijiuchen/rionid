# RionID+ (Ringed IONs IDentification)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8169341.svg)](https://doi.org/10.5281/zenodo.8169341)

**RionID+** 是一个用于储存环 Schottky 频谱分析的开源软件。它从实验频谱中识别粒子种类，通过模拟粒子在环中的回旋频率，将实验峰与来自 LISE 文件或 AME 数据库的已知核素进行匹配，从而确定离子的质量电荷比（m/q）和产额。

> Forked from [DFreireF/rionid](https://github.com/DFreireF/rionid) with extensive enhancements for GUI-based interactive analysis.

---

## 目录

- [程序结构](#程序结构)
- [架构概览 (MVC)](#架构概览-mvc)
- [模块说明](#模块说明)
- [流程图](#流程图)
- [参数说明](#参数说明)
- [运行示例](#运行示例)
- [SMS mode 自动扫描](#sms-mode-自动扫描)
- [阈值 Profile 交互编辑](#阈值-profile-交互编辑)
- [支持的文件格式](#支持的文件格式)

---

## 程序结构

```
rionid/
├── setup.py                  # 安装脚本
├── setup.cfg                 # 包元数据
├── pyproject.toml            # 构建配置
├── parameters_cache.toml     # 参数缓存文件
├── README.md                 # 本文档
├── rionid/                   # 核心模型层 (Model)
│   ├── __init__.py           # 导出核心类
│   ├── __main__.py           # CLI 入口
│   ├── version.py            # 版本 (5.2.1)
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
    └── parameter_gui.py       # 参数面板 (RionID_GUI)
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
│  │  │ Run /         │  │  │  │  阈值线 / 基线      │  │  │
│  │  │ SMS    │  │  │  │  峰标记 (红三角)    │  │  │
│  │  │ Exit         │  │  │  │  图例 / 光标位置    │  │  │
│  │  └───────────────┘  │  │  └─────────────────────┘  │  │
│  │        Controller   │  │         View              │  │
│  └─────────┬───────────┘  └───────────────────────────┘  │
└────────────┼──────────────────────────────────────────────┘
             │ visualization_signal
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
│  │  TDMS/    │  │          │  │ (轻量扫描)            │  │
│  │  ROOT/XML)│  │          │  │                       │  │
│  └───────────┘  └──────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**数据流：**

```
[实验数据文件] → ImportData._get_experimental_data()
                → detect_peaks_and_widths() → 峰列表

[LISE 粒子文件] → _set_particles_to_simulate_from_file()
                → _calculate_moqs() → 粒子数据库

[用户参数] → _calculate_srrf(fref/brho/ke)
            → _simulated_data(harmonics) → simulated_data_dict

[匹配] → compute_matches(threshold, freq_min, freq_max)
       → (χ², match_count, matched_ions)
```

---

## 模块说明

### `rionid/importdata.py` — 核心模型

| 方法 | 功能 |
|------|------|
| `__init__` | 初始化参数，加载实验数据 |
| `_parse_ion_name("205Tl81+")` | 解析离子名 → (质量数, 元素, 电荷) |
| `_get_experimental_data(filename)` | 根据扩展名自动识别格式并导入数据 |
| `detect_peaks_and_widths()` | 峰检测（基于阈值 profile 或固定比例） |
| `_set_particles_to_simulate_from_file(filep)` | 读取 LISE 格式粒子文件 |
| `_calculate_moqs()` | 计算每个离子的 m/q 比值 |
| `_calculate_srrf(fref/brho/ke)` | 计算相对回旋频率 srrf |
| `_simulated_data(harmonics, mode)` | 模拟各谐波的测量频率和产额 |
| `compute_matches(threshold, ...)` | 将实验峰与模拟频率匹配，返回 χ² 和 match_count |
| `scan_match(f_ref, alphap, ...)` | **轻量扫描**：复用粒子缓存，只重算 srrf 和频率（优化 SMS） |
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

在 **Kinetic Energy 模式**下，通过每核子动能计算 γ：

```
γ = (KE × A) / m + 1
```

然后同样通过 γ → β → v → f_rev 计算回旋频率。

χ² 匹配使用产额加权：

```
χ² = Σ(weight_i × Δf_i²) / Σ(weight_i)
```

### `rionid/pyqtgraphgui.py` — 可视化 (View)

| 组件 | 功能 |
|------|------|
| `CreatePyGUI` | 主绘图窗口 (QMainWindow) |
| `plot_experimental_data(data)` | 绘制蓝色实验谱线，可选基线和基线去除曲线 |
| `plot_simulated_data(data)` | 绘制各谐波的离子峰竖线和标签 |
| `red_triangles` | 检测到的峰标记（红色三角） |
| `threshold_profile_line/points` | 阈值 profile 曲线和编辑点 |
| `mouse_moved(evt)` | 实时显示光标位置频率和幅度 |
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
| `run_script()` | "Run" 按钮：执行单次模拟 + 匹配 |
| `SMS_pid_script()` | "Run SMS"：自动扫描 f_ref 和 alphap |
| `enterPlotPickMode()` | 进入图谱点击取频率模式 |
| `_update_harmonic_calculation()` | 自动计算基频和各谐波频率 |

### `rionidgui/gui_controller.py` — 控制器

| 函数 | 功能 |
|------|------|
| `import_controller(...)` | 串联所有计算步骤，返回 ImportData 对象 |
| `display_nions(nions, ...)` | 按产额排序截断显示指定数量的离子 |
| `save_simulation_results(...)` | 保存模拟结果到文件 |

---

## 流程图

```
                    ┌───────────────┐
                    │   MainWindow  │
                    │  (gui.py)     │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
     ┌────────▼────────┐       ┌─────────▼────────┐
     │  Parameter Panel │       │  PyQtGraph Plot  │
     │ (parameter_gui)  │       │ (pyqtgraphgui)   │
     │                  │       │                  │
     │ Mode / αp / f_ref│       │  实验数据 (蓝)    │
     │ Harmonics / 离子  │       │  模拟峰竖线 (彩色)│
     │ Run / SMS  │       │  峰标记 (红三角) │
     │ Pick from plot   │       │  阈值线 / 基线    │
     └────────┬─────────┘       └──────────────────┘
              │
              │ visualization_signal / overlay_sim_signal
              ▼
     ┌─────────────────┐
     │ import_controller│
     │ (gui_controller)  │
     └────────┬─────────┘
              │
     ┌────────▼─────────────────────────────────┐
     │            ImportData                     │
     │  (rionid/importdata.py)                   │
     │                                           │
     │  1. 加载实验数据                           │
     │     ├─ .npz / .csv / .tdms / .root / .xml │
     │     └─ detect_peaks_and_widths() → 峰列表  │
     │                                           │
     │  2. 加载 LISE 粒子文件 → _calculate_moqs() │
     │                                           │
     │  3. _calculate_srrf(fref/brho)            │
     │     ├─ srrf[i] = 1 - αp × Δmoq/moq_ref    │
     │     └─ ref_frequency 确定                  │
     │                                           │
     │  4. _simulated_data(harmonics)             │
     │     ├─ 各谐波频率 = srrf × f₀ × h          │
     │     └─ simulated_data_dict[谐波]            │
     │                                           │
     │  5. compute_matches(threshold)             │
     │     ├─ 实验峰 ↔ 模拟频率匹配               │
     │     ├─ χ² = Σ(w·Δf²)/Σ(w)                │
     │     └─ match_count + matched_ions          │
     └───────────────────────────────────────────┘
```

---

## 参数说明

### 文件选择区

| 参数 | 说明 |
|------|------|
| **Experimental Data File** | 实验频谱数据文件（支持多种格式） |
| **.lpp File** | LISE 格式粒子列表文件（含各核素产额） |

### 数据处理选项

| 参数 | 说明 |
|------|------|
| **Reload Experimental Data** | 是否重新加载实验数据（取消则使用缓存） |
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

### 输出文件

| 参数 | 说明 |
|------|------|
| **Simulation result** | 模拟结果保存路径 |
| **matched result** | 详细匹配结果 CSV 保存路径 |

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

1. **选择实验数据文件** — 点击 Browse 选择 `.npz` / `.csv` / `.tdms` / `.root` 等
2. **选择 LISE 粒子文件** — 点击 Browse 选择 `.lpp` 文件
3. **设置模式** — 通常选 `Frequency`
4. **输入参考频率** — 在 `Value` 输入参考离子回旋频率（Hz）
5. **输入参考谐波数** — 在 `Ref. Harmonic` 输入参考频率对应的谐波数
6. **输入参考离子** — 如 `86Kr25+`
7. **输入谐波列表** — 如 `124 125 126`
8. **输入 αp** — 动量压缩因子（如 0.08）
9. **设置匹配阈值** — 如 `500` Hz
10. 点击 **Run** 执行模拟与匹配

### 3. SMS mode 自动扫描

自动扫描参考频率和 αp 的最佳组合：

1. 在 **SMS mode Settings** 区域设置扫描范围：
   - αp min / max / step（如 `0.07 0.09 0.0005`）
   - Reference frequency min / max（如 `3.08e8 3.12e8`）
2. 点击 **Run SMS**
3. 程序自动遍历所有组合，按 match_count 降序 / χ² 升序选择最佳参数

**优化说明：** SMS 内层循环使用 `scan_match()` 轻量扫描方法，**只重算 srrf 和频率**，复用粒子数据缓存，避免每轮重建 ImportData 对象。扫描结果会自动绘制最佳匹配的频谱。

### 4. CLI 命令行模式

```bash
# Frequency 模式
python -m rionid data.npz -f 15686564.45 -r 80Kr35+ -psim particles.lpp -ap 0.0546 -hrm 124 125 126 -s -o output_folder

# Bρ 模式
python -m rionid data.npz -b 5.8494 -r 80Kr35+ -psim particles.lpp -ap 0.0546 -hrm 1 -s
```

### 5. 输出文件说明

| 文件 | 内容 |
|------|------|
| `simulation_result.out` | 各谐波的模拟结果（离子名、频率、产额、m/q、质量） |
| `best_match_details.csv` | 匹配详细信息（匹配频率、峰宽、γₜ、回旋周期等） |
| `{filename}_cache.npz` | 实验数据缓存（加速下次加载） |
| `{filename}_height_thresh.csv` | 频率相关阈值 profile（交互编辑保存） |

---

## SMS mode 自动扫描

**Run SMS** 是自动寻找最佳物理参数的功能，其内部逻辑为：

### 扫描算法

```
for each f_ref in exp_peaks_hz_filtering:      # 外层：遍历实验峰作为参考频率
    for each test_alphap in range(alphap_min, alphap_max, step):  # 内层：扫描 αp
        chi2, match_count = data.scan_match(f_ref, test_alphap, ...)
        results.append((f_ref, test_alphap, chi2, match_count))

# 选出最佳组合
sorted_results = sorted(results, key=lambda x: (-x[3], x[2]))
# 先按 match_count 降序，同匹配数时按 χ² 升序
best_fref, best_alphap, best_chi2, best_match_count = sorted_results[0]

# 用最佳参数运行完整模拟并绘图
```

### 优化设计

普通模式每次迭代都重建 `ImportData`（含文件 I/O）。SMS 做了以下优化：

- **外围一次性构建** `ImportData` 对象，加载粒子数据
- **内层使用 `scan_match()`** — 只重算 `srrf` 和频率，无文件 I/O

---

## 阈值 Profile 交互编辑

程序支持交互式编辑频率依赖的峰检测阈值：

1. 在 **Threshold Profile** 行点击 **Browse** 选择 `height_thresh.csv`（或手动输入路径）
2. 点击 **Start Click Threshold** 按钮（变绿色）
3. 在频谱图上 **左键点击** — 在点击位置添加/更新一个绿色星形阈值点
4. **右键**已有阈值点 — 弹出菜单选择 "Delete threshold point"
5. 阈值点自动保存到 CSV，并触发峰重检测和重绘

**阈值应用逻辑：** 峰检测时，阈值不再全局固定，而是使用频率线性插值得到每个频率点的局部阈值。

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

当前版本：**5.2.1**

## 致谢

- Dr. RuiJiu Chen ([@chenruijiu](https://github.com/chenruijiu/)) — 原始 C++ ToF 模拟代码
- Dr. Shahab Sanajri ([@xaratustrah](https://github.com/xaratustrah/)) — 软件架构指导
- G. Hudson-Chang ([@gwgwhc](https://github.com/gwgwhc/)) — LISEreader 和教程
