## 一、自相关 (Autocorrelation)

### 1.1 定义与公式

**自相关函数**描述一个信号与其自身在不同时间延迟下的相似程度。对于连续时间信号 $x(t)$，其自相关函数定义为：

$$R_{xx}(\tau) = \int_{-\infty}^{\infty} x(t)x(t+\tau)dt$$

对于离散时间信号 $x[n]$，自相关函数为：

$$R_{xx}[m] = \sum_{n=-\infty}^{\infty} x[n]x[n+m]$$

其中 $\tau$ 或 $m$ 为时间延迟量。

### 1.2 主要性质

1. **偶函数性质**：$R_{xx}(\tau) = R_{xx}(-\tau)$，图形对称于纵轴
2. **最大值性质**：当 $\tau=0$ 时，$R_{xx}(0) \geq R_{xx}(\tau)$，且 $R_{xx}(0)$ 等于信号的能量
3. **周期性**：周期信号的自相关函数也是周期函数，且周期与原信号相同
4. **极限性质**：对于能量信号，$\lim_{\tau \to \infty} R_{xx}(\tau) = 0$

### 1.3 物理意义

自相关函数反映了信号在不同时间延迟下的自我相似性。当信号中存在周期性分量时，自相关函数会在相应的周期位置出现峰值，这一特性可用于从噪声中检测周期性信号。

<span class="image main">
<img class="main img-in-blog" style="max-width: 50%" src="./blogs/18_Autocorrelation_Cross-correlation_Convolution/Effects_of_autocorrelation.webp" alt="Effects of autocorrelation" />
<i>Above: A plot of a series of 100 random numbers concealing a sine function. <br>Below: The sine function revealed in a correlogram produced by autocorrelation.<br>By <a href="//commons.wikimedia.org/wiki/File:Acf.svg" title="File:Acf.svg">Acf.svg</a>: Jeremy Manningderivative work: <a href="//commons.wikimedia.org/w/index.php?title=User:Jrmanning&amp;action=edit&amp;redlink=1" class="new" title="User:Jrmanning (page does not exist)">Jrmanning</a> (<a href="//commons.wikimedia.org/wiki/User_talk:Jrmanning" title="User talk:Jrmanning"><span class="signature-talk">talk</span></a>) - <a href="//commons.wikimedia.org/wiki/File:Acf.svg" title="File:Acf.svg">Acf.svg</a>, <a href="http://creativecommons.org/licenses/by-sa/3.0/" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=7172158">Link</a></i>
</span>

## 二、互相关 (Cross-correlation)

### 2.1 定义与公式

**互相关函数**描述两个不同信号之间的相似程度。对于连续时间信号 $x(t)$ 和 $y(t)$，互相关函数定义为：

$$R_{xy}(\tau) = \int_{-\infty}^{\infty} x(t)y(t+\tau)dt$$

对于离散时间信号 $x[n]$ 和 $y[n]$，互相关函数为：

$$R_{xy}[m] = \sum_{n=-\infty}^{\infty} x[n]y[n+m]$$

### 2.2 主要性质

1. **非对称性**：$R_{xy}(\tau) \neq R_{yx}(\tau)$，但满足 $R_{xy}(\tau) = R_{yx}(-\tau)$
2. **有界性**：$|R_{xy}(\tau)| \leq \sqrt{R_{xx}(0)R_{yy}(0)}$
3. **极限性质**：$\lim_{\tau \to \infty} R_{xy}(\tau) = 0$（对于能量信号）

### 2.3 物理意义

互相关函数用于衡量两个信号之间的相似性。当两个信号形状相似但存在时间延迟时，互相关函数会在该延迟位置出现峰值。这一特性广泛应用于信号检测、时延估计和模式匹配等领域。

## 三、卷积 (Convolution)

### 3.1 定义与公式

**卷积**是信号处理中的基本运算，描述一个信号对另一个信号的“作用”。对于连续时间信号 $f(t)$ 和 $g(t)$，卷积定义为：

$$(f\ast g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau$$

对于离散时间信号 $x[n]$ 和 $h[n]$，卷积为：

$$y[n] = (x\ast h)[n] = \sum_{k=-\infty}^{\infty} x[k]h[n-k]$$

### 3.2 主要性质

1. **交换律**：$f\ast g = g\ast f$
2. **结合律**：$(f\ast g)\ast h = f\ast (g\ast h)$
3. **分配律**：$f\ast (g+h) = f\ast g + f\ast h$
4. **线性性**：$a(f\ast g) = (af)\ast g = f\ast (ag)$
5. **时移不变性**：若 $f(t) \to f(t-t_0)$，则 $(f\ast g)(t) \to (f\ast g)(t-t_0)$

### 3.3 物理意义

卷积在系统分析中具有重要物理意义：若 $x(t)$ 为系统输入，$h(t)$ 为系统冲激响应，则输出 $y(t) = x(t)*h(t)$。这表示系统对任意输入的响应可以通过输入与冲激响应的卷积得到。

## 四、三者之间的关系

<span class="image main">
<img class="main img-in-blog" style="max-width: 60%" src="./blogs/18_Autocorrelation_Cross-correlation_Convolution/Comparison_convolution_correlation.webp" alt="Comparison convolution and correlation" />
<i>自相关、互相关与卷积的比较, By <a href="//commons.wikimedia.org/wiki/User:Cmglee" title="User:Cmglee">Cmglee</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/3.0" title="Creative Commons Attribution-Share Alike 3.0">CC BY-SA 3.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=20206883">Link</a></i>
</span>

### 4.1 数学关系

**互相关与卷积的关系**：

$$R_{xy}(\tau) = x(\tau)*y(-\tau)$$

**自相关与卷积的关系**：

$$R_{xx}(\tau) = x(\tau)*x(-\tau)$$

**卷积与互相关的区别**：
- 卷积需要对其中一个信号进行翻转（$g(t-\tau)$）
- 互相关不需要翻转（$g(t+\tau)$）
- 当卷积核为偶函数时，卷积与互相关结果相同

### 4.2 运算过程对比

| 运算类型 | 是否翻转 | 交换律 | 物理意义 |
|---------|---------|--------|---------|
| 卷积 | 需要翻转 | 满足 | 系统响应 |
| 互相关 | 不需要翻转 | 不满足 | 相似性度量 |
| 自相关 | 不需要翻转 | 满足 | 自我相似性 |

## 五、应用场景

### 5.1 自相关的应用

- **信号周期性检测**：从噪声中提取周期性信号
- **雷达信号处理**：通过自相关函数确定脉冲重复间隔(PRI)
- **语音处理**：音高检测和降噪
- **设备故障诊断**：检测旋转机械的周期性故障

### 5.2 互相关的应用

- **时延估计**：测量信号传播时间，如雷达测距、声纳定位
- **信号检测**：在噪声中检测已知波形（匹配滤波）
- **图像匹配**：在图像处理中进行特征匹配
- **多通道信号处理**：估计到达时间差

### 5.3 卷积的应用

- **线性滤波**：实现低通、高通、带通等滤波器
- **系统响应计算**：计算线性时不变系统对任意输入的响应
- **图像处理**：实现模糊、锐化、边缘检测等操作
- **神经网络**：卷积神经网络(CNN)中的特征提取

## 六、MATLAB实现示例

### 6.1 自相关计算

```matlab
% 计算信号x的自相关
[R_xx, lag] = xcorr(x, x);
plot(lag, R_xx);
title('信号自相关函数');
```

### 6.2 互相关计算

```matlab
% 计算信号x和y的互相关
[R_xy, lag] = xcorr(x, y);
plot(lag, R_xy);
title('信号互相关函数');
```

### 6.3 卷积计算

```matlab
% 计算信号x和h的卷积
y = conv(x, h);
plot(y);
title('卷积结果');
```

## 七、总结

自相关、互相关和卷积是信号处理中的核心概念，三者既有密切联系又有本质区别。自相关用于分析信号的自我相似性和周期性，互相关用于测量两个信号之间的相似性和时间延迟，卷积则用于描述系统对输入的响应。在实际应用中，三者常结合使用，构成信号处理和分析的基础工具集。