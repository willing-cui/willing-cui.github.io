## 问题建模

首先对待研究的问题建立数学模型
在[一阶倒立摆的系统模型](index.html?part=blogs&id=0)这篇文章里，我们已经做了完整的受力分析。最终得到了关于系统变量的微分方程。

$$ (M+m)\ddot{x}+b\dot{x}-ml\ddot{\psi}= u$$
$$(I+ml^2)\ddot{\psi}-mgl\psi = ml\ddot{x}$$

## 状态空间

可以将状态空间理解为一个包含系统输入、系统输出和状态变量的集合，它们之间的关系可以用一个一阶微分方程表达出来。

<div class="formula-in-blog">

$$ 状态空间（集合）=\left\\{
\begin{aligned}
系统输入\\\\
系统输出\\\\
状态变量
\end{aligned}
\right\\}=一阶微分方程
$$

</div>

通过观察我们知道，目前系统模型是以二阶微分方程的形式描述的。为了消除方程中的高阶项，可以整理得到下面的式子。

<div class="formula-in-blog">

$$\left\\{\begin{array}{l}
\dot{x}=\dot{x}\\\\
\ddot{x}=\frac{m^2l^2g}{I(m+M)+mMl^2} \psi-\frac{b(I+ml^2)}{I(m+M)+mMl^2} \dot{x}+\frac{(I+ml^2)}{I(m+M)+mMl^2}u\\\\
\dot{\psi}=\dot{\psi}\\\\
\ddot{\psi}=\frac{mlg(m+M)}{I(m+M)+mMl^2}\psi-\frac{mlb}{I(m+M)+mMl^2}\dot{x}+\frac{ml}{I(m+M)+mMl^2}u
\end{array}\right.$$

</div>

已知系统状态空间方程的标准形式为：

<div class="formula-in-blog">

$$
\left\\{\begin{array}{l}
\dot{x}=Ax+Bu\\\\
y=Cx+Du
\end{array}\right.
$$

</div>

式中$\dot{x}$表示系统中的一阶微分项，$y$表示系统状态。以矩阵运算的形式表示系统的状态空间方程：

<div class="formula-in-blog">

$$\begin{aligned}
\begin{bmatrix}
\dot{x}\\\\
\ddot{x}\\\\
\dot{\psi}\\\\
\ddot{\psi}
\end{bmatrix}=
&\begin{bmatrix}
0 & 1 & 0 & 0\\\\
0 & -\frac{b(I+ml^2)}{I(m+M)+mMl^2} & \frac{m^2l^2g}{I(m+M)+mMl^2} & 0\\\\
0 & 0 & 0 & 1 \\\\
0 & -\frac{mlb}{I(m+M)+mMl^2} & \frac{mlg(m+M)}{I(m+M)+mMl^2} & 0
\end{bmatrix} \times
\begin{bmatrix}
x\\\\
\dot{x}\\\\
\psi\\\\
\dot{\psi}
\end{bmatrix} \\\\
&+
\begin{bmatrix}
0\\\\
\frac{(I+ml^2)}{I(m+M)+mMl^2}\\\\
0\\\\
\frac{ml}{I(m+M)+mMl^2}
\end{bmatrix}
u
\end{aligned}$$

</div>

<div class="formula-in-blog">

$$
y= \begin{bmatrix}
1 & 0 & 0 & 0\\\\
0 & 0 & 1 & 0
\end{bmatrix} \times
\begin{bmatrix}
x\\\\
\dot{x}\\\\
\psi\\\\
\dot{\psi}
\end{bmatrix} +
\begin{bmatrix}
0 \\\\
\end{bmatrix} \times
u
$$

</div>

## LQR控制器设计

L（Linear）Q（Quadratic）R（Regulator），直译为线性二次型控制器。
可以通过加入反馈，使系统最终能达到稳定状态，但如何选取最好的系统特征值（极点）在系统收敛的前提下实现最优的收敛过程，这是我们目前要考虑的问题。
这里我们引入目标函数（价值函数），使系统在收敛的同时，满足$J$最小：

<div class="formula-in-blog">

$$
J=\int ^{t_f}_{t_0}[X^TQX+U^TRU]dt\\\\
min(J)
$$

</div>

1. $Q$是一个对角矩阵，$X^TQX=ax_1^2+bx_2^2+cx_3^2+......$，当系统中的状态变量$x≠0$时，我们可以通过调节$Q$中元素的值来改变该变量的对$J$的影响。$Q$中较大的一项对应在收敛过程中优先考虑的系统变量。
2. 同理，$U^TRU$则代表了系统输入$U$对价值函数$J$的影响。因为$J$是积分的形式，当矩阵$R$中某一项较大，则意味着我们希望该项对应的系统输入能够快速收敛到0。这么做的现实意义往往是以最小的代价（例如能耗）实现系统的稳态。
<font color='red'>式子中代表系统输入的$U$在一个能够实现自稳定的系统中（例如我们这里设计的倒单摆系统）代表控制器反馈回路的输出。</font>

我们目前涉及的倒立摆系统只有一个输入，即倒立摆所受到的外部牵引力$U$，所以矩阵$R$仅有一个元素。另外有四个系统状态变量，分别是$x, \dot{x}, \psi, \dot{\psi}$，因此对角矩阵$Q$的规模为$4\times4$。

在MATLAB中，我们可以调用$lqr()$函数生成满足$J$最小的反馈矩阵$K$。$K$对应的就是各个系统变量反馈路径中的增益。即：

<div class="formula-in-blog">

$$U=[K_1, K_2, K_3, K_4]\times\begin{bmatrix}
x\\\\
\dot{x}\\\\
\psi\\\\
\dot{\psi}
\end{bmatrix}$$

</div>

下图是倒立摆开环系统的阶跃响应，显然系统是不收敛的。

<span class="image main">

<img class="main img-in-blog" style="max-width: 40rem" src="./blogs/2_对一阶倒立摆的LQR控制/open_loop.webp" alt="开环阶跃响应" />

</span>

引入LQR反馈后系统的阶跃响应。

<span class="image main">

<img class="main img-in-blog" style="max-width: 40rem" src="./blogs/2_对一阶倒立摆的LQR控制/LQR_control.webp" alt="LQR阶跃响应" />

</span>

## MATLAB 代码

```bash
clc; 
clear; 
close all;

% Parameters:
%   m: mass of pendulum (kg)
%   M: mass of cart (kg)
%   b: dampling coefficient
%   I: rotional inertia
%   g: acceleration of gravity
%   L: the distance from mass center to the hinge

m = 3.375;
M = 5.40;
b = 0.01;
I = 0.0703125;
g = 9.80665;
L = 0.25;

%  create transfer function model
s = tf('s');

q = (m + M) * (I + m * L^2) - (m * L)^2;

P_cart = (((I + m * L^2) / q) * s^2 - (m * g * L / q)) / ...
    (s^4 + (b * (I + m * L^2)) * s^3 / q - ((M + m) * m * g * L) * s^2 / q - b * m * g * L * s / q);

P_pend = (m * L * s / q) / ...
    (s^3 + (b * (I + m * L^2)) * s^2 / q - ((M + m) * m * g * L) * s / q - b * m * g * L / q);

sys_tf = [P_cart; P_pend];

inputs = {'u'}; 
outputs = {'x'; 'phi'};
set(sys_tf,'InputName',inputs);
set(sys_tf,'OutputName',outputs);

sys_tf

% create state-space model
p = I * (m + M) + m * M * L^2;

A = [0, 1, 0, 0;
     0, -b * (I + m * L^2) / p, (m^2 * L^2 * g) / p, 0;
     0, 0, 0, 1;
     0, -(m * L * b) / p, m * L * g * (M + m) / p, 0];
 
B = [0;
     (I + m * L^2) / p;
     0;
     m * L / p];

C = [1, 0, 0, 0;
     0, 0, 1, 0];
 
D = [0;
     0];

% definitions of system variables
states = {'x' 'x_dot' 'phi' 'phi_dot'};
inputs = {'u'}; outputs = {'x'; 'phi'};

sys_ss = ss(A, B, C, D, 'statename', states, 'inputname', inputs, 'outputname', outputs);

% poles of open loop system
poles = pole(sys_tf);

poles

figure();
% impulse response of system
t = 0: 0.01: 1;
impulse(sys_ss, t);

figure();
% step response of system
t = 0: 0.01: 1;
step(sys_ss, t);


% LQR simulation
% Q matrix of LQR controller
Q = [1000, 0, 0, 0;
     0, 0, 0, 0;
     0, 0, 500, 0;
     0, 0, 0, 0];
 
R = 0.1;

% optimal gain matrix K
K = lqr(A, B, Q, R);

Ac = A - B * K;
sys_lqr = ss(Ac, B, C, D, 'statename', states, 'inputname', inputs, 'outputname', outputs);

figure();
% impulse response of system
t = 0: 0.01: 2;
impulse(sys_lqr, t);

figure();
% step response of system
t = 0: 0.01: 3;
step(sys_lqr, t);
```
