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
使用梯形法 `np.trapz` 分别在峰区间和本底区间积分：

```
peak_area = ∫_{li}^{ri} amp(f) df
bg_area   = ∫_{bg_li}^{bg_ri} amp(f) df
net_area  = peak_area - bg_area
```

### 4.5 每帧平均面积
为消除投影时间窗口长度不同的影响，将净面积除以帧数：

```
A_per_frame = net_area / n_frames
```

这也是 projection 图上显示的 `A/f=...` 数值的含义。

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

## 6. 归一化面积 vs 频率（Plot Norm Area）

### 6.1 归一化
对每个 projection，以该投影的**第一个峰**（或最后一个峰）的面积为参考，对**该投影内所有峰的净面积**做归一化：

```
norm_area_ij = net_area_ij / net_area_{i, ref}
```

其中 `i` 为投影序号，`j` 为峰序号，`ref` 为第一个或最后一个峰（由 `Norm: 1st peak` / `Norm: last peak` 下拉框选择）。

### 6.2 共振曲线拟合
将所有投影的归一化面积与频率一起绘图，用 **Lorentzian 型共振曲线**拟合：

```
R(f) = A₀ / [1 + Q² × (f/f_sys - f_sys/f)²]
```

其中：
- **`A₀`** — 共振峰顶幅度
- **`Q`** — 品质因子（Q 越大峰越窄）
- **`f_sys`** — 共振中心频率（MHz）

拟合使用 ROOT 的 `TF1` + `TGraph.Fit`（最小二乘法），得到 A₀、Q、f_sys 及其拟合误差。

> **注意**：此处"归一化面积"使用的是**未扣除本底**的原始峰面积 `pk['areas']`。而 Self-consistent 迭代中的归一化面积使用**扣除本底后的净面积**（`pk['areas'] - pk['bg_levels']`），因此两者得到的共振参数可能有微小差异。统一使用扣除本底的数据进行共振拟合是推荐做法——即使用 Self-consistent 迭代（或后续版本中更新 Plot Norm Area 的逻辑）。

### 6.3 残差图
下子图显示 data - fit 残差，用于判断拟合质量。

### 6.4 对话框
按钮 **Plot Norm Area** 打开一个独立对话框，包含：
- A₀ / Q / f_sys 参数输入框（默认值来自上一次拟合结果）
- **Fit & Plot** 按钮（重新拟合并更新图形）
- 上方：归一化面积 + 拟合曲线（对数纵轴）
- 下方：残差散点图

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

拟合使用 ROOT TF1，得到：

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

## 11. 参数持久化（Config Save/Load）

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

## 12. 参数参考表

| UI 控件 | 参数 | 默认值 | 说明 |
|---------|------|--------|------|
| `Proj: [5] + [20] s` | offset, dt | 5, 20 | 电压事件后投影时间窗口（s） |
| `dist(MHz):` | min_distance | 0.003 | 最小峰间距（MHz） |
| `width:` | integration_width | 0.001 | 峰和本底的积分宽度（MHz） |
| `gap:` | bg_gap | 0.001 | 峰区间与本底区间的间距（MHz） |
| `l` | baseline_l | 100 | BrPLS 平滑度 |
| `ratio` | baseline_ratio | 0.001 | BrPLS 比例参数 |
| `A₀` | resonance amplitude | — | 共振曲线幅度 |
| `Q` | quality factor | — | 共振品质因子 |
| `f_sys` | resonance freq | — | 共振中心频率（MHz） |
| `h_off` | harmonic offset | 0 | 全局谐波偏移 |
| `per:` | per-projection offset | "" | 逐投影谐波偏移 |
| `Norm:` | norm reference | 1st peak | 归一化参考峰位置 |

---

## 13. 完整操作流程示例

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
