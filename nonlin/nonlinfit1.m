clc
clear

%% DATA
% Read Data
data = readmatrix('hoorn_0.1rad_constant_80s.xlsx');
time = data(:,1);

xr1 = data(:,2); % r(k)
xr2 = data(:,3); % r(k-1)
xr3 = data(:,4); % -r(k)+r(k-1)
xr4 = data(:,5); % r(k-1)
xr5 = data(:,7); % del(k-1)
xr6 = data(:,8); % del(k)-del(k-1)
x = [xr1 xr2 xr3 xr4 xr5 xr6];
y = data(:,9); % r(k+1)

% xr1 = data(:,4); % -r(k)+r(k-1)
% xr2 = data(:,5); % r(k-1)
% xr3 = data(:,7); % del(k-1)
% xr4 = data(:,8); % del(k)-del(k-1)
% x = [xr1 xr2 xr3 xr4];
% y = data(:,10); % r(k+1)-2*r(k)+r(k-1)

% nninfit: https://www.mathworks.com/help/stats/nlinfit.html#btk7ign-X
modelfun = @(b,x)(2*xr1 -xr2+b(1)*xr3+b(2)*xr4+b(3)*xr5+b(4)*xr6);
beta0 = [0;0;0;0];
beta = nlinfit(x,y,modelfun,beta0)

%% Find T1,T2,K,T3
% vpa: https://www.mathworks.com/help/symbolic/vpa.html
% solve: https://www.mathworks.com/help/symbolic/sym.solve.html
h = 0.01;
syms T1 T2 K T3
eqn1 = (-h^2)/(T1*T2) == beta(2);
sT1 = vpa(simplify(solve(eqn1,T1)),7)

eqn2 = (sT1+T2)*h/(sT1*T2) == beta(1);
sT2 = vpa(solve(eqn2,T2),7)

eqn3 = (-h^2)/(T1*sT2(1)) == beta(2);
sT1 = vpa(simplify(solve(eqn3,T1)),7)

eqn4 = (h^2)*K/(sT1*sT2(1)) == beta(3);
sK = vpa(simplify(solve(eqn4,K)),7)

eqn5 = h*sK*T3/(sT1*sT2(1)) == beta(4);
sT3 = vpa(simplify(solve(eqn5,T3)),7)

%% TF coefficient
a1 = double(sT3*sK) 
a2 = double(sK)

b1 = double(sT1*sT2(1))
b2 = double(sT1+sT2(1))
b3 = 1

%% TF Plot 
num = [a1 a2];
dem = [b1 b2 b3];
sys = tf(num,dem)

% clf
t = 0:0.01:50;
% u = 0.1;
lsim(sys,data(:,6),t)

