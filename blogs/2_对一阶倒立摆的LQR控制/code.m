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