## 1. 核心概念

**自相关函数** 是一种衡量一个信号与它自身在不同时间点上的相似度（相关性）的函数。它回答了一个关键问题：“给定当前时刻的信号，我能多大程度上预测未来某一时刻（即经过一个**时间滞后 $\tau$**）的信号？”

-   **高自相关**：如果信号在时间滞后 $\tau$ 后与自身高度相似，说明信号在该时间尺度上变化缓慢，具有“记忆性”。
-   **低自相关**：如果信号在很短的时间滞后后就变得不相似，说明信号变化很快，几乎是随机的。

在Wi-Fi CSI的语境下，ACF用于分析CSI时间序列内部的统计规律。

## 2. 定义与计算

对于从单个子载波上采集到的CSI时间序列 $H(f, t)$，其自相关函数 $\rho_H(f, \tau)$ 定义为归一化的协方差：

$$
\rho_{H}(f,\tau) \triangleq \frac{\mathrm{Cov}[H(f,t), H(f,t+\tau)]}{\mathrm{Cov}[H(f,t), H(f,t)]}
$$

其中：
-   $\mathrm{Cov}[\cdot,\cdot]$ 表示协方差。
-   $H(f, t)$ 是频率为 $f$ 的子载波在时刻 $t$ 的信道状态信息（一个复数值）。
-   $\tau$ 是**时间滞后**。
-   分母是信号在零滞后时的方差，用于将ACF归一化到 $[-1, 1]$ 的范围内，便于比较。

## 3. 在散射模型中的应用与速度估计

在充满散射体的复杂环境中，基于散射模型，可以推导出ACF与动态散射体速度 $v$ 之间的定量关系：

$$
\rho_{H}(f,\tau) \approx \mathrm{sinc}(kv\tau) = \frac{\sin(kv\tau)}{kv\tau}
$$

其中 $k = 2\pi / \lambda$ 是波数（$\lambda$ 为信号波长）。

<span class="image main">
<img class="main img-in-blog" style="max-width: 50%" src="./blogs/17_Autocorrelation_Function/sinc_fun.webp" alt="Sinc Function" />
<i>sinc函数, By <a href="//commons.wikimedia.org/wiki/User:Georg-Johann" title="User:Georg-Johann">Georg-Johann</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/3.0" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=17007237">Link</a></i>
</span>

### 速度提取方法

通过匹配测量到的ACF与理论的 $\mathrm{sinc}$ 函数，可以计算出速度 $v$：

1.  **测量ACF**：计算CSI时间序列的ACF，找到其**第一个峰值**所对应的时间滞后 $\tau_0$。
2.  **匹配理论模型**：已知 $\mathrm{sinc}(x)$ 函数的第一个峰值出现在固定位置 $x_0 \approx 1.43\pi\approx4.49$。因此有 $k v \tau_0 = x_0$。
3.  **计算速度**：速度 $v$ 可解为：
    $$
    v = \frac{x_0}{k \tau_0} = \frac{x_0 \lambda}{2\pi \tau_0}
    $$

## 4. ACF揭示的其他关键统计特性

通过分析ACF在不同维度上的行为，可以提取出更多信道参数：

#### a) 频率相关性分析
-   **相干带宽**：ACF在频率轴上衰减到某一阈值时所对应的频率宽度。衰减越慢，相干带宽越大，说明信道频率响应越平坦（多径效应不显著）；衰减越快，相干带宽越小，表明**时延扩展**越大（多径效应显著）。

#### b) 空间相关性分析（适用于多天线系统）
-   **相干距离**：ACF在空间轴上衰减到阈值时所对应的天线间距。这决定了MIMO系统中获得独立空间信道所需的天线间距。

## 5. 总结

自相关函数（ACF）是将原始、嘈杂的CSI“快照”转化为具有明确物理意义的、稳定的“统计特征”的关键桥梁。它通过量化信道的“记忆”效应，为速度估计、活动识别等Wi-Fi感知应用提供了鲁棒且有效的特征提取方法。