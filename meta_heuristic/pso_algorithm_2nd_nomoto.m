clc
clear

%% DATA
% Read Data

data = readmatrix('hoorn_0dot1rad_50s_0dot0sample.xlsx');
dsize = 4500;
data = data(1:dsize,:);
time = data(:,1);

deltadot = data(:,10);
r = data(:,15);
delta = data(:,21);

%% Model Formulation
% Construct state matrix
rback = [r(1);r(1:(length(r)-1))]; % Get r(k-1)
deltaback = [delta(1);delta(1:(length(delta)-1))]; % Get delta(k-1)

xr1 = r; % r(k)
xr2 = rback; % r(k-1)
xr3 = -r+rback; % -r(k)+r(k-1)
xr4 = rback; % r(k-1)
xr5 = deltaback; % del(k-1)
xr6 = delta-deltaback; % del(k)-del(k-1)
x = [xr1 xr2 xr3 xr4 xr5 xr6];

% Output
y = [r(2:length(time)); r(length(time))];

%% PSO
pop_size = 1000;
param_size = 4;

%{
Step 1 – Initialisation:
Firstly, we creates a random population with a defined number of particles 
(potential solutions). Each solution includes a random position and speed.
%}

pos_min = -1;
pos_max = 1;
pop_pos = (pos_max - pos_min).*rand(pop_size,param_size)+pos_min;

vel_min = -0.1;
vel_max = 0.1;
pop_vel = (vel_max - vel_min).*rand(pop_size,param_size)+vel_min;

pbest_fitness = Inf(pop_size,1);
pbest_param = zeros(pop_size,param_size);

gbest_fitness = Inf;
gbest_param = zeros(1,param_size);

iter=1000;
w = 0.7;                % inertia coefficient
c1 = 1.2;               % cognitive parameter (0 <= c1 <= 2)
c2 = 1.2;               % social parameter (0 <= c2 <= 2)
r_min = 0;              % for r1 & r2
r_max = 1;

for i = 1:iter
    
    % Step 2 – Evaluation
    for j = 1:pop_size
        err_sum = 0;
        for k = 1:dsize
            pred = 2*x(k,1) - x(k,2)+pop_pos(j,1)*x(k,3)-...
            pop_pos(j,2)*x(k,4)+pop_pos(j,3)*x(k,5)+pop_pos(j,4)*x(k,6);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        
        % Step 3.1 - Assigning pbest
        if err_sum < pbest_fitness(j,1)
            pbest_fitness(j,1) = err_sum;
            pbest_param(j,:) = pop_pos(j,:);
        end
        
        % Step 3.2 - Assigning gbest
        if min(pbest_fitness) < gbest_fitness
            [gbest_fitness,min_i] = min(pbest_fitness);
            gbest_param(1,:) = pbest_param(min_i,:);
        end
    end
    
    % Step 4.1 - Update speed
    for j = 1:pop_size
        for h = 1:param_size
            r1 = (r_max - r_min)*rand(1) + r_min;
            r2 = (r_max - r_min)*rand(1) + r_min;
            pop_vel(j,h) = w*pop_vel(j,h) + c1*r1*(pbest_param(j,h) - pop_vel(j,h)) + c2*r2*(gbest_param(1,h) - pop_vel(j,h));
        end
    end

    % Step 4.2 - Update pos
    for j = 1:pop_size
        for h = 1:param_size
            pop_pos(j,h) = pop_pos(j,h) + pop_vel(j,h);
        end
    end
end

beta = gbest_param(1,:);

%% Find T1,T2,K,T3
% vpa: https://www.mathworks.com/help/symbolic/vpa.html
% solve: https://www.mathworks.com/help/symbolic/sym.solve.html
h = 0.05;
syms T1 T2 K T3
eqn1 = (h^2)/(T1*T2) == beta(2);
sT1 = vpa(simplify(solve(eqn1,T1)),2);

eqn2 = (sT1+T2)*h/(sT1*T2) == beta(1);
sT2 = vpa(simplify(solve(eqn2,T2)),2);

eqn3 = (h^2)/(T1*sT2(1)) == beta(2);
sT1 = vpa(simplify(solve(eqn3,T1)),2);

eqn4 = (h^2)*K/(sT1*sT2(1)) == beta(3);
sK = vpa(simplify(solve(eqn4,K)),2);

eqn5 = h*sK*T3/(sT1*sT2(1)) == beta(4);
sT3 = vpa(simplify(solve(eqn5,T3)),7);

%% TF coefficient
a1 = double(sT3*sK);
a2 = double(sK);

b1 = double(sT1*sT2(1));
b2 = double(sT1+sT2(1));
% b2 = sT1+sT2(1)
b3 = 1;

%% Discrete Plot
T1 = double(sT1)
T2 = double(sT2(1))
K = double(sK)
T3 = double(sT3)
b1 = T1*T2
b2 = T1 + T2

ynew = zeros(length(time),1);
for i = 1:(length(time) - 1)
   ynew(i+1) = 2*xr1(i) - xr2(i) + ((T1+T2)*h/(T1*T2))*xr3(i) - ((h^2)/(T1*T2))*xr4(i) + (K*(h^2)/(T1*T2))*xr5(i) + (K*T3*h/(T1*T2))*xr6(i);
end

figure
hold on
plot(time,ynew);
plot(time,r);
legend('legend','r');
hold off

%% Other
% Time series for delta
delta_ts = timeseries(delta,time);
deltad_ts = timeseries(deltadot,time);
r_ts = timeseries(r,time);

%% Fitness Functions

function err = ise(pred,meas)
    err = (pred - meas)^2;
end
