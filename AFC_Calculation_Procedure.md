# AFC (Amplifier Function Calculation) 计算过程说明

## 概述

AFC（Amplifier Function Calculation，放大器函数计算）的核心目标是**从 2D 频谱图中提取特定束流的峰面积随时间的变化**，并去除共振腔响应函数（即 Amplifier Function）的调制效应。

**算法的核心思想**：实验测得的峰面积 $A_{\text{raw}}(f, t)$ 同时受到共振放大 $R(f)$ 和离子数衰减 $e^{-\lambda t}$ 的影响。由于同一投影内各谐波的峰共享相同的衰减因子，归一化可以消除衰减项，使面积仅由 $R(f)$ 决定。但 $R(f)$ 本身未知——这是一个**自洽问题**：要得到准确的 $R(f)$ 需要已知的未调制面积，而要得到未调制面积又需要已知的 $R(f)$。

解决方案是**自洽迭代**：
1. 用初始猜测的 $R(f)$ 修正各投影面积（AF 修正）
2. 用修正后的面积重新拟合 $R(f)$
3. 重复 1–2 直到 $R(f)$ 收敛

收敛后，用最终的 $R(f)$ 修正目标谐波面积，拟合指数衰变曲线得到 $\lambda$ 和 $T_{1/2}$。

完整流程包括：频谱投影 → 本底扣除 → 寻峰与积分 → 谐波分配 → 共振曲线拟合 → AF 修正 → 自洽迭代 → 衰变曲线拟合。

---

## 1. 输入数据

### 1.1 频谱文件（Experiment Time File）
2D 频谱图（spectrogram）：横轴为频率（MHz），纵轴为时间（s），颜色深度表示信号强度。支持的格式：
- `.root`（TH2 直方图）
- `.npz`（缓存的 numpy 数组，自动生成）
- `.csv` / `.txt` 等

### 1.2 电压文件（Voltage File）
记录实验中**电压跳变事件**的时间点。文件至少有两列：
- 列 0：电压值（V）
- 列 1：事件发生时间（s）

每个电压事件定义一个注入时刻 `t_v`，用于后续的频谱投影提取。

---

## 2. 频谱投影（Projection）

对于每个电压事件在时刻 `t_v`，提取 **时间窗口** `[t_v + offset, t_v + offset + dt]` 内的频谱切片，沿时间轴求和得到该电压下的 1D 频率谱（projection）。

```
Projection_i(f) = Σ_{t ∈ [t_v + offset, t_v + offset + dt]} S(t, f)
```

其中 `offset` 和 `dt` 由用户在面板上的 `Proj: [offset] + [dt] s` 文本框设定（默认 offset = 5 s, dt = 20 s）。

同时记录该 projection 包含的频谱帧数 `n_frames = len(idx)`，用于后续的每帧平均面积归一化。

---

## 3. 本底扣除（Remove Baseline）

使用 **BrPLS**（Baseline removal by Penalized Least Squares）算法对每个 1D projection 扣除基线：

```
amp_clean(f) = amp_raw(f) - baseline(f)
```

参数由 `l`（平滑度，默认 100）和 `ratio`（默认 0.001）控制。扣除后：
- `self._projections` — 原始投影
- `self._projections_baseline` — 基线
- `self._projections_clean` — 扣除基线后的投影

如图中显示：灰色 = 原始谱，橙色 = 基线，蓝色 = 扣除后谱。

---

## 4. 峰检测（Find Peaks）

在基线扣除后的频谱中检测峰，参数：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `dist(MHz)` | 最小峰间距（MHz） | 0.003 |
| `width` | 峰和本底的积分宽度（MHz） | 0.001 |
| `gap` | 峰区间与本底区间的间距（MHz） | 0.001 |

### 4.1 寻峰算法
使用 `scipy.signal.find_peaks`，阈值由 threshold profile 文件或峰高百分比控制。

### 4.2 积分区间
以每个峰的**峰值位置 `p`** 为中心，取宽度为 `width`（MHz）的对称区间：

```
half_bins = round(width / 2 / df)
li = p - half_bins
ri = p + half_bins
```

其中 `df = freq[1] - freq[0]` 为频率分辨率（MHz/bin）。

### 4.3 本底区间
本底区间与峰区间**宽度相同**，放在峰左侧，间距为 `gap`（MHz）：

```
bg_width_bins = half_bins * 2
gap_bins = round(gap / df)
bg_li = li - bg_width_bins - gap_bins
bg_ri = bg_li + bg_width_bins
```

若左侧超出频谱范围则自动改放右侧。

### 4.4 面积计算

峰面积分为**总投影面积**和**逐帧面积统计**两部分。

#### 4.4.1 总投影面积（用于报告峰值）

使用梯形法 `np.trapz` 在**总投影（所有帧求和后）**的峰区间和本底区间上积分：

```
peak_area_total   = trapz(amp[li:ri+1], freq[li:ri+1])
bg_area_total     = trapz(amp[bg_li:bg_ri+1], freq[bg_li:bg_ri+1])
```

其中 `peak_area_total` 保存为 `pk['areas'][j]`，`bg_area_total` 保存为 `pk['bg_levels'][j]`。

#### 4.4.2 逐帧面积与不确定度（用于误差传播）

当程序存储了逐帧频谱数据（`self._projections_frame_data`）时，对**每一帧**分别计算峰积分和本底积分，然后统计净面积在帧间波动：

```
For each frame k:
    pk_k = trapz(frame_amp[k, li:ri+1], freq[li:ri+1])
    bg_k = trapz(frame_amp[k, bg_li:bg_ri+1], freq[bg_li:bg_ri+1])
    net_k = pk_k - bg_k

peak_areas_err[j] = std(net_1, net_2, ..., net_N) * sqrt(N)    (N > 1 且 std > 0)
```

误差取帧间净面积的标准差（ddof=1）乘以 √N，保存在 `pk['areas_err'][j]` 中。

> **为什么需要逐帧误差？** 传统的泊松误差（√计数）只反映统计涨落的期望值，而逐帧误差能捕获**实际测量的帧间波动**——包括系统噪声、本底不稳定性和谱线形状变化等真实不确定性来源。这与 `AFC_npzmonitor.ipynb` 中使用的泊松近似 `δR = R·√(1/A_i+1/A_base)` 思路一致，但误差来源来自实验数据而非统计模型。

### 4.5 每帧平均面积

为消除投影时间窗口长度不同的影响，可将净面积除以帧数：

```
A_per_frame = net_area / n_frames
```

这也是 projection 图上显示的 `A/f=...` 数值的含义。
Area vs Time 和 Self-consistent 中使用的面积也除以了 `n_frames` 以进行归一化。

---

## 5. 谐波分配（Fit Har）

对每个 projection，将检测到的峰按频率大小排序，线性拟合谐波编号：

```
f(n) = n × f₀ + offset
```

其中 `n` 为谐波编号（0, 1, 2, ...），`f₀` 为基频，`offset` 为频率偏移。用户可设定：

- **`h_off`** — 全局谐波偏移，如 `2` 则谐波编号整体 +2
- **`per`** — 逐投影偏移，如 `0:3 2:-1` 表示 projection 0 额外 +3，projection 2 额外 -1

---

## 6. 面积比 vs 频率（Plot Norm Area）

### 6.1 频对构建与面积比

Plot Norm Area 使用 **面积比方法**（与 `AFC_npzmonitor.ipynb` 一致），而非直接拟合归一化面积。

对每个投影，根据 **Ref:** 下拉框选择参考峰：
- **Ref: 1st peak** — 该投影的第一个峰
- **Ref: last peak** — 该投影的最后一个峰
- **Ref: Max Area** — 该投影中面积最大的峰

以参考峰的面积 $A_{\mathrm{ref}}$ 为分母，该投影内其他各峰 $A_i$ 为分子，构建频对 $(f_i, f_{\mathrm{ref}})$ 和面积比：

$$R_i = \frac{A_i}{A_{\mathrm{ref}}}$$

面积比的不确定度来自面积误差的传播：

$$\delta R_i = R_i \cdot \sqrt{\left(\frac{\delta A_i}{A_i}\right)^2 + \left(\frac{\delta A_{\mathrm{ref}}}{A_{\mathrm{ref}}}\right)^2}$$

其中 $\delta A$ 来自逐帧净面积波动（`pk['areas_err']`，见 4.4.2 节）。

### 6.2 共振曲线拟合

使用 **洛伦兹谐振响应函数的面积比模型** 进行拟合。

#### 谐振响应函数

$$H(f, Q, f_{\mathrm{sys}}) = \frac{1}{\sqrt{1 + Q^2 \left( \dfrac{f}{f_{\mathrm{sys}}} - \dfrac{f_{\mathrm{sys}}}{f} \right)^2}}$$

其中：
- **`Q`** — 品质因子（Q 越大峰越窄）
- **`f_sys`** — 共振中心频率（MHz）

#### 理论面积比

实验测量的峰面积 $A(f) \propto |H(f)|^2$，因此两个频率的峰面积之比为：

$$R_{\mathrm{theory}}(f_1, f_2, Q, f_{\mathrm{sys}}) = \frac{|H(f_1)|^2}{|H(f_2)|^2}
= \frac{1 + Q^2 \left( \dfrac{f_2}{f_{\mathrm{sys}}} - \dfrac{f_{\mathrm{sys}}}{f_2} \right)^2}
       {1 + Q^2 \left( \dfrac{f_1}{f_{\mathrm{sys}}} - \dfrac{f_{\mathrm{sys}}}{f_1} \right)^2}$$

#### 加权最小二乘拟合

寻找参数 $(Q, f_{\mathrm{sys}})$ 最小化加权残差平方和：

$$\chi^2(Q, f_{\mathrm{sys}}) = \sum_{i=1}^{N} \left( \frac{R_{\mathrm{exp}}^{(i)} - R_{\mathrm{theory}}(f_1^{(i)}, f_2^{(i)}, Q, f_{\mathrm{sys}})}{\delta R^{(i)}} \right)^2$$

使用 `scipy.optimize.curve_fit`，参数边界：

$$Q \in [10, 20000], \qquad f_{\mathrm{sys}} \in [0.9 \cdot f_{\mathrm{median}}, 1.1 \cdot f_{\mathrm{median}}]$$

其中 $f_{\mathrm{median}}$ 为所有检测峰中心频率的中值。

#### 蒙特卡洛模拟（5000 次）

为更真实地估计参数不确定度，执行 5000 次蒙特卡洛模拟：

$$R_{\mathrm{sim}}^{(k)} = R_{\mathrm{exp}} + \epsilon^{(k)}, \quad \epsilon^{(k)} \sim \mathcal{N}(0, \delta R)$$

每次使用 `curve_fit` 重新拟合，得到 $(Q^{(k)}, f_{\mathrm{sys}}^{(k)})$。最终结果取均值和标准差：

$$\begin{aligned}
Q_{\mathrm{final}} &= \mathrm{mean}(Q_{\mathrm{samples}}) \pm \mathrm{std}(Q_{\mathrm{samples}}) \\
f_{\mathrm{sys, final}} &= \mathrm{mean}(f_{\mathrm{samples}}) \pm \mathrm{std}(f_{\mathrm{samples}})
\end{aligned}$$

蒙特卡洛方法比 `curve_fit` 的线性近似更稳健，能捕获非线性效应和参数边界效应。

### 6.3 输出

#### 图形（2×3 布局）
- **上子图（跨 3 列）**：面积比散点图（带误差棒）+ 拟合共振曲线
- **左下子图**：残差（data - fit，带误差棒）
- **中下子图**：Q 的 MC 分布直方图
- **右下子图**：f_sys 的 MC 分布直方图

图中文本框显示两种方法的结果对比：
```
Least Squares:  Q = 10415 ⨦ 152    f_sys = 308.125000 ⨦ 4.50e-02 MHz
MC (4998 iter): Q = 10415 ⨦ 168    f_sys = 308.125000 ⨦ 5.10e-02 MHz
```

#### CSV 输出
输出至 `afc_norm_area.csv`，包含频对、面积比和拟合残差。

> **与旧方法的区别**：旧方法将同一投影内各峰分别除以参考峰面积得到归一化面积，用 ROOT TF1 拟合模型 $A_0 / [1 + Q^2(f/f_{\mathrm{sys}} - f_{\mathrm{sys}}/f)^2]$。新方法直接拟合**面积比** $R = A_i / A_{\mathrm{ref}}$，使用比值模型 $|H(f_1)|^2 / |H(f_2)|^2$，避免了 $A_0$ 参数的拟合，并加入了逐帧误差传播和蒙特卡洛不确定度估计。

---

## 7. AF 修正（Amplitude Factor Correction）

### 7.1 修正原理
AF 修正用于**消除共振响应函数对各投影面积幅度的调制**。每个投影的面积乘以一个因子，使其等效于在参考频率 `f_ref` 处测量的值：

```
AF_i = R(f_i_ref) / R(f_ref)        # 投影 i 相对于 f_ref 的放大倍数比
corrected_area_ij = norm_area_ij × AF_i
```

其中 `f_i_ref` 为投影 i 的参考峰频率，`f_ref` 为第一个投影的参考峰频率。

### 7.2 AF_i 的频率来源
- **`f_ref`** — 第一个 projection 的参考峰频率（根据归一化选择：第一个峰或最后一个峰）
- **`f_i_ref`** — 每个投影 i 的参考峰频率（与归一化使用同样的索引）

### 7.3 使用场景

#### Plot Norm Area
仅展示归一化面积 vs 频率的共振拟合，**不**做 AF 修正。

#### Self-consistent 迭代
在每次迭代中先用当前共振曲线做 AF 修正，再用修正后的面积重新拟合共振曲线，如此反复直到收敛。

---

## 8. 自洽迭代（Self-consistent）

点击 **Self-consistent** 按钮打开新面板，允许用户手动逐次迭代：

### 8.1 迭代流程

每次点击 **Iterate (1 step)** 执行一次迭代：

```
Step a) 用当前 AF 修正后的面积拟合共振曲线 R(f) → 新 (A₀, Q, f_sys)
Step b) 用新参数计算 AF 因子 → 修正各投影面积
         corrected_area = norm_area × (AF_i / AF_ref)
         AF_i = R(f_first_i),  AF_ref = R(f_ref)
Step c) 更新图形，显示修正后的面积 + 拟合曲线
```

最多迭代 20 次。收敛判据（自动模式下）：Q 相对变化 < 0.5% 且 f_sys 相对变化 < 1 ppm。

### 8.2 界面布局
```
┌───────────────────────────────────────┐
│  A0: [...]  Q: [...]  f_sys: [...]    │ ← 可手动编辑初始参数
│  [Iterate (1 step)]  [Fit Decay]       │ ← 两个按钮
│  Iter 5: A₀=... Q=... f_sys=...       │ ← 状态信息
├───────────────────────────────────────┤
│  ┌───────────────────────────────────┐│
│  │  上图：AF 修正后归一化面积 vs 频率   ││
│  │  (对数纵轴，带拟合参数文本框)        ││
│  ├───────────────────────────────────┤│
│  │  下图：残差 / 衰减曲线              ││
│  │  默认：残差散点图                   ││
│  │  点击 Fit Decay：切换到衰减曲线     ││
│  └───────────────────────────────────┘│
└───────────────────────────────────────┘
```

---

## 9. 衰变曲线拟合（Fit Decay）

迭代收敛后（或直接点击 **Fit Decay**），用最终的共振参数 `(A₀, Q, f_sys)` 修正目标谐波的面积，拟合指数衰减：

### 9.1 修正目标谐波面积

```
R_factor_i = R(f_har_i)                   # 第 i 个投影中目标谐波的共振放大倍数
corrected_area_i = net_area_i / R_factor_i
corrected_area_i = corrected_area_i / corrected_area_0   # 归一化到第一点为 1
```

### 9.2 指数衰减拟合

```
A(t) = exp(-λ × t)
```

拟合使用 `scipy.optimize.curve_fit`（加权最小二乘，`absolute_sigma=True`），以面积误差作为权重，得到：

- **λ** — 衰变常数（s⁻¹），显示为 `λ = 1.23e-03 ± 1.45e-05 s⁻¹`
- **T₁/₂** — 半衰期（s），显示为 `T₁/₂ = 563.45 ± 6.67 s`

计算公式：

```
λ = fit_parameter
T₁/₂ = ln(2) / λ
σ(T₁/₂) = T₁/₂ × σ(λ) / λ
```

### 9.3 图形
衰减曲线下方显示带误差的参数文本框（绿色背景）。

---

## 10. 谐波 - 时间面积曲线（Area vs Time）

点击 **Area vs Time** 按钮，对指定谐波绘制其面积随时间的变化：

### 10.1 数据提取
从每个 projection 中提取目标谐波号对应的峰，计算：

```
raw_area = peak_area - bg_area                # 净面积
raw_area_per_frame = raw_area / n_frames       # 每帧平均面积
```

### 10.2 曲线
- **蓝色曲线** — 原始每帧平均净面积 vs 时间
- **粉色曲线** — 修正后面积 vs 时间：`corrected = raw / R(freq)`，归一化到第一点为 1
- **绿色曲线** — 指数拟合：`A(t) = exp(-λ × t)`

### 10.3 参数控制
A₀、Q、f_sys 参数可手动调节（自动填入上一次共振拟合的结果），点击 **Update** 更新图形。

---

## 11. 输出文件说明

AFC Calculator 在操作过程中会在当前工作目录生成以下输出文件：

### 11.1 `afc_peaks.csv` — 峰检测结果

点击 **Find Peaks** 时生成。每行对应一个检测到的峰，每列为：

| 列名 | 说明 | 单位 |
|------|------|------|
| `time_s` | 投影的时间标签 | s |
| `voltage_V` | 对应的电压值 | V |
| `frequency_MHz` | 峰中心频率 | MHz |
| `height` | 峰高（幅度） | arb. unit |
| `FWHM_MHz` | 半高全宽 | MHz |
| `area` | 峰面积（积分区间内的原始面积，**未扣除本底**） | arb. unit |
| `area_err` | 峰面积不确定度（逐帧净面积标准差 × √N） | arb. unit |
| `mean` | 峰内加权平均频率（质心频率） | MHz |
| `std` | 峰内频率分布标准差 | MHz |

> **注意**：该文件在每次点击 Find Peaks 时被覆盖。`area` 列是原始积分面积（总投影求和后积分），`area_err` 来自逐帧净面积的帧间波动。如需扣除本底的净面积需从 `afc_harmonics.csv` 或代码内部数据结构中获取。

### 11.2 `afc_harmonics.csv` — 谐波分配结果

点击 **Fit Har** 时生成。在 `afc_peaks.csv` 基础上增加谐波编号和拟合信息：

| 列名 | 说明 | 单位 |
|------|------|------|
| `proj_idx` | 投影序号 | — |
| `time_s` | 投影时间 | s |
| `voltage_V` | 电压值 | V |
| `harmonic_n` | 分配的谐波编号（0 = 未匹配） | — |
| `freq_MHz` | 峰频率 | MHz |
| `height` | 峰高 | arb. unit |
| `FWHM_MHz` | 半高全宽 | MHz |
| `area` | 峰面积 | arb. unit |
| `area_err` | 峰面积不确定度 | arb. unit |
| `mean_freq_MHz` | 质心频率 | MHz |
| `std_freq_MHz` | 频率分布标准差 | MHz |
| `f0_fit_MHz` | 拟合基频 | MHz |
| `offset_fit_MHz` | 频率偏移（线性拟合截距） | MHz |
| `residual_MHz` | 拟合残差 | MHz |
| `kept` | 是否参与拟合（1 = 是，0 = 否） | — |

该文件每次点击 Fit Har 时被覆盖。最后一行以 `#` 开头为投影拟合摘要（f0、offset、R²、谐波数）。

### 11.3 `afc_norm_area.csv` — 面积比与共振拟合

点击 **Plot Norm Area** 时自动生成。

| 列名 | 说明 | 单位 |
|------|------|------|
| `proj_idx` | 投影序号 | — |
| `time_s` | 投影时间 | s |
| `voltage_V` | 电压值 | V |
| `f1_MHz` | 分子峰频率 | MHz |
| `f_ref_MHz` | 分母（参考）峰频率 | MHz |
| `area_ratio` | 面积比 $R = A_i / A_{\mathrm{ref}}$ | — |
| `delta_ratio` | 面积比不确定度 $\delta R$ | — |
| `fitted_ratio` | 拟合面积比 $R_{\mathrm{theory}}(f_1, f_2)$ | — |
| `residual` | $R_{\mathrm{exp}} - R_{\mathrm{theory}}$ | — |

### 11.4 `afc_calculator_config.json` — 参数配置文件

程序自动保存和读取，包含所有面板控件的当前值（详见第 12 节参数参考表）。用户可删除此文件恢复所有默认参数。

---

## 12. 参数持久化（Config Save/Load）

所有面板参数自动保存到 `afc_calculator_config.json`，在每次点击按钮时触发保存。重启程序后自动读取恢复。

保存的参数包括：
- 文件路径（频谱文件、电压文件、threshold 路径）
- 投影参数（offset, dt, split ratio）
- 寻峰参数（dist, width, gap）
- 本底参数（l, ratio）
- 谐波参数（h_off, per）
- 归一化参考峰选择（1st / last peak）
- 目标谐波编号
- 显示选项（Log Z, Show Threshold, Show Projections）
- 字体缩放倍数
- 上次共振拟合参数（A₀, Q, f_sys）

---

## 13. 参数参考表

| UI 控件 | 参数 | 默认值 | 说明 |
|---------|------|--------|------|
| `Proj: [5] + [20] s` | offset, dt | 5, 20 | 电压事件后投影时间窗口（s） |
| `dist(MHz):` | min_distance | 0.003 | 最小峰间距（MHz） |
| `width:` | integration_width | 0.001 | 峰和本底的积分宽度（MHz） |
| `gap:` | bg_gap | 0.001 | 峰区间与本底区间的间距（MHz） |
| `l` | baseline_l | 100 | BrPLS 平滑度 |
| `ratio` | baseline_ratio | 0.001 | BrPLS 比例参数 |
| `Q` | quality factor | — | 共振品质因子 |
| `f_sys` | resonance freq | — | 共振中心频率（MHz） |
| `h_off` | harmonic offset | 0 | 全局谐波偏移 |
| `per:` | per-projection offset | "" | 逐投影谐波偏移 |
| `Ref:` | norm reference | Max Area | 参考峰选择（1st / last / Max Area） |

---

## 14. 完整操作流程示例

```
1. Load Data & Plot       加载频谱文件 + 电压文件
2. Project                计算频谱投影（可调 offset/dt）
3. Remove Baseline        扣除本底（可调 l/ratio）
4. Find Peaks             检测峰（可调 dist/width/gap）
5. Fit Har                分配谐波编号（可调 h_off/per）
6. Plot Norm Area         拟合共振曲线，查看归一化面积
7. Self-consistent        自行迭代 AF 修正 + 共振拟合
8.   Iterate (1 step)     手动逐次迭代
9.   Fit Decay            拟合衰变曲线 → λ, T₁/₂
10. Area vs Time          查看特定谐波的面积-时间曲线
```

---

## 附录：共振函数公式

### 探测系统的共振响应模型

探测系统的共振响应函数采用 Lorentzian 型模型：

$$R(f) = \frac{A_0}{1 + Q^2 \left(\frac{f}{f_{\text{sys}}} - \frac{f_{\text{sys}}}{f}\right)^2}$$

这是一个对称的峰状函数，在 $f = f_{\text{sys}}$ 处取最大值 $A_0$。Q 值越大，共振峰越窄。

**考虑离子衰变的完整模型**

实际测量中，峰面积同时受共振响应和离子数量衰减的影响。离子数随时间呈指数衰减：

$$N(t) = N_0 \cdot e^{-\lambda t}$$

因此，在时刻 $t$、频率 $f$ 处测得的原始峰面积为：

$$A_{\text{raw}}(f, t) = R(f) \cdot N_0 \cdot e^{-\lambda t} = \frac{A_0 N_0 \cdot e^{-\lambda t}}{1 + Q^2 \left(\frac{f}{f_{\text{sys}}} - \frac{f_{\text{sys}}}{f}\right)^2}$$

由于每个投影 $i$ 的所有峰在**同一时刻 $t_i$** 测量，它们共享相同的衰减因子 $e^{-\lambda t_i}$。归一化步骤（除以该投影参考峰的面积）恰好消去这个共享的衰减因子，使得归一化面积只由共振函数 $R(f)$ 决定：

$$A_{i,j}^{\text{(norm)}} = \frac{A_{\text{raw}}(f_{i,j}, t_i)}{A_{\text{raw}}(f_{i,\text{ref}}, t_i)} = \frac{R(f_{i,j})}{R(f_{i,\text{ref}})}$$

这就是为什么来自不同时间投影的归一化面积可以放在一起拟合同一条共振曲线 $R(f)$——衰减效应已被归一化消除。

### 自洽迭代计算共振曲线的过程

自洽迭代的核心思路是：**用共振曲线修正面积 → 用修正后的面积重新拟合共振曲线 → 重复直到收敛**。每一步的数学描述如下。

---

#### 符号定义

| 符号 | 含义 |
|------|------|
| $R(f)$ | 共振响应函数（Lorentzian 型） |
| $A_{i,j}^{\text{(raw)}}$ | 第 $i$ 个投影、第 $j$ 个峰的原始峰面积 |
| $B_{i,j}$ | 对应的本底面积 |
| $A_{i,j}^{\text{(net)}}$ | 净面积：$A_{i,j}^{\text{(net)}} = A_{i,j}^{\text{(raw)}} - B_{i,j}$ |
| $A_{i,j}^{\text{(norm)}}$ | 每个投影内归一化的净面积 |
| $f_{i,j}$ | 第 $i$ 个投影、第 $j$ 个峰的频率 |
| $n_i$ | 第 $i$ 个投影包含的频谱帧数 |
| $N$ | 投影总数 |
| $M_i$ | 第 $i$ 个投影的峰数 |

---

#### 步骤 0：归一化与初始化

对每个投影 $i$，取该投影的参考峰（第一个或最后一个，由用户选择），将其净面积作为归一化基准：

$$A_{i}^{\text{(ref)}} = A_{i,\text{ref}}^{\text{(net)}}$$

对投影 $i$ 内的所有峰，做归一化：

$$A_{i,j}^{\text{(norm)}} = \frac{A_{i,j}^{\text{(net)}}}{A_{i}^{\text{(ref)}}}$$

**为什么要做这个归一化？**

每个投影的多个峰来自同一束离子的不同谐波。由于离子的数量随时间指数衰减：

$$N(t) = N_0 \cdot e^{-\lambda t}$$

因此，同一投影中不同谐波的峰都受到相同倍数的时间衰减因子 $N(t_i)/N_0$ 的影响。**归一化本质上除以了这个共同的衰减因子**，使得归一化后的面积只反映共振腔的频率响应和各谐波本身的强度比，而与离子的衰减无关。

这样做的好处是，不同投影、不同谐波的归一化面积可以直接放在一起拟合同一条共振曲线 $R(f)$——因为面积随频率的变化只由共振响应决定，不再受衰减影响。

以第一个投影的参考峰频率作为参考频率：

$$f_{\text{ref}} = f_{0,\text{ref}} = \begin{cases}
f_{0,0} & \text{(Norm: 1st peak)}\\
f_{0,-1} & \text{(Norm: last peak)}
\end{cases}$$

使用上一次保存的拟合结果（或默认值）作为初始参数 $A_0^{(0)}, Q^{(0)}, f_{\text{sys}}^{(0)}$。

---

#### 迭代循环

##### Step a：拟合共振曲线

将当前修正后的归一化面积作为数据点，使用 ROOT TF1 拟合共振函数：

$$\{f_{i,j},\; \tilde{A}_{i,j}^{(k-1)}\} \xrightarrow{\text{least squares fit}} \{A_0^{(k)}, Q^{(k)}, f_{\text{sys}}^{(k)}\}$$

其中拟合模型为：

$$R^{(k)}(f) = \frac{A_0^{(k)}}{1 + [Q^{(k)}]^2 \left(\frac{f}{f_{\text{sys}}^{(k)}} - \frac{f_{\text{sys}}^{(k)}}{f}\right)^2}$$

拟合过程最小化：

$$\chi^2 = \sum_{i=0}^{N-1} \sum_{j=0}^{M_i-1} \left[\tilde{A}_{i,j}^{(k-1)} - R^{(k)}(f_{i,j})\right]^2$$

##### Step b：AF 因子修正

用新得到的共振参数计算每个投影的 AF（放大倍数）因子。

先计算每个投影的参考峰的共振放大倍数：

$$R(f_{i,\text{ref}}) = \frac{A_0^{(k)}}{1 + [Q^{(k)}]^2 \left(\frac{f_{i,\text{ref}}}{f_{\text{sys}}^{(k)}} - \frac{f_{\text{sys}}^{(k)}}{f_{i,\text{ref}}}\right)^2}$$

AF 因子定义为当前投影参考峰的放大倍数与全局参考频率 $f_{\text{ref}}$ 处的放大倍数之比：

$$\text{AF}_i = \frac{R(f_{i,\text{ref}})}{R(f_{\text{ref}})}$$

对投影 $i$ 中的所有峰，乘以同一个 AF 因子：

$$\tilde{A}_{i,j}^{(k)} = A_{i,j}^{\text{(norm)}} \times \text{AF}_i = A_{i,j}^{\text{(norm)}} \times \frac{R^{(k)}(f_{i,\text{ref}})}{R^{(k)}(f_{\text{ref}})}$$

这里 $A_{i,j}^{\text{(norm)}}$ 是原始归一化面积，在迭代过程中**保持不变**——每次迭代都是从原始归一化面积出发，用新的 AF 因子重新修正。

---

##### 为什么要做 AF 因子修正？（物理背景）

探测系统（共振腔）的响应在不同频率处的放大倍数不同：

$$R(f) = \frac{A_0}{1 + Q^2 \left(\frac{f}{f_{\text{sys}}} - \frac{f_{\text{sys}}}{f}\right)^2}$$

- 在共振频率 $f_{\text{sys}}$ 处，信号被最大放大（幅度 = $A_0$）
- 偏离 $f_{\text{sys}}$ 时，放大幅度迅速下降

这意味着：**即使两个投影中核素的真实产额相等，只要它们的频率不同，测得的峰面积也会不同**。这就是共振调制效应——我们想要消除它。

**为什么选第一个投影的第一个峰作为参考 $f_{\text{ref}}$？**

需要一个统一的基准。所有投影的面积都变换到「如果它们是在同一个参考频率 $f_{\text{ref}}$ 处测量」会是多少，这样投影之间的差异才真正反映产额的变化。

**为什么投影 $i$ 内所有峰共用一个因子？**

同一个投影内的所有峰来自**同一个电压事件、同一段时间窗口**，它们同时被测量，共享完全相同的共振腔响应条件。因此它们应该统一修正。

**AF 因子的物理含义**

$$\text{AF}_i = \frac{R(f_{i,\text{ref}})}{R(f_{\text{ref}})}$$

- 如果投影 $i$ 参考峰的频率离 $f_{\text{sys}}$ 较近 → $R(f_{i,\text{ref}}) > R(f_{\text{ref}})$ → $\text{AF}_i > 1$ → 该投影所有峰的面积放大
- 如果投影 $i$ 参考峰的频率离 $f_{\text{sys}}$ 较远 → $R(f_{i,\text{ref}}) < R(f_{\text{ref}})$ → $\text{AF}_i < 1$ → 该投影所有峰的面积缩小

修正后的面积 $\tilde{A}_{i,j} = A_{i,j}^{\text{(norm)}} \times \text{AF}_i$ 等效于「所有投影都在同一参考频率 $f_{\text{ref}}$ 处测量」时的面积。这样，投影之间的面积差异就只来自核素产额的真实变化，而不再包含共振响应的贡献。

---

#### 收敛判据

当拟合参数的相对变化同时满足以下条件时，认为迭代已收敛：

$$\left|\frac{Q^{(k)} - Q^{(k-1)}}{Q^{(k-1)}}\right| < 0.5\% \quad \text{且} \quad \left|\frac{f_{\text{sys}}^{(k)} - f_{\text{sys}}^{(k-1)}}{f_{\text{sys}}^{(k-1)}}\right| < 1 \times 10^{-6}$$

即 Q 的相对变化 < 0.5%，且 f_sys 的相对变化 < 1 ppm。

---

#### 收敛后的衰变曲线拟合

迭代收敛后，用最终的共振参数 $A_0, Q, f_{\text{sys}}$ 修正目标谐波面积。对每个投影中目标谐波 $n_{\text{tar}}$ 对应的峰：

$$A_i^{\text{(corr)}} = \frac{A_i^{\text{(net)}}}{R(f_{i, n_{\text{tar}}})}$$

归一化到第一点：

$$\hat{A}_i = \frac{A_i^{\text{(corr)}}}{A_0^{\text{(corr)}}}$$

拟合单参数指数衰减模型：

$$\hat{A}(t) = \exp(-\lambda \cdot t)$$

其中时间以第一个数据点为原点（$t_0$ 为第一点的时间）：

$$t_i' = t_i - t_0$$

拟合得到衰变常数 $\lambda$，进而得到半衰期：

$$T_{1/2} = \frac{\ln 2}{\lambda}, \quad \sigma(T_{1/2}) = T_{1/2} \cdot \frac{\sigma(\lambda)}{\lambda}$$

---

### 指数衰减函数

$$A(t) = \exp(-\lambda t)$$

半衰期与衰变常数的关系：

$$T_{1/2} = \frac{\ln 2}{\lambda}$$
