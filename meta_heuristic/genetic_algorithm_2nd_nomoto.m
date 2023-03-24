clc
clear

%% DATA
% Read Data

% data = readmatrix('hoorn_0dot1rad_50s_0dot0sample.xlsx');
data = readmatrix('hoorn_15deg_15hold_100s_0dot05sample.xlsx');
dsize = 1500;
data = data(1:dsize,:);
time = data(:,1);
% udot = data(:,2);
% vdot = data(:,3);
% rdot = data(:,4);
% xdot = data(:,5);
% ydot = data(:,6);
% psidot = data(:,7);
% pdot = data(:,8);
% phidot = data(:,9);
deltadot = data(:,10);
% n1dot = data(:,11);
% n2dot = data(:,12);
% u = data(:,13);
% v = data(:,14);
r = data(:,15);
% x = data(:,16);
% y = data(:,17);
% psi = data(:,18);
% p = data(:,19);
% phi = data(:,20);
delta = data(:,21);
% n1 = data(:,22);
% n2 = data(:,23);
% delta_cmd = data(:,24);

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

%% Genetic Algorithm
pop_size = 10000;
param_size = 4;
% Chuong Nguyen Khanh - Novel Adaptive Control Method for BLCD Drive of 
% ElectricBike for Vietnam EnvironmenT - IASF-2019
% doi:10.1088/1757-899X/819/1/012017

%{
Step 1 – Initialisation:
GA initializes a random population with a defined amount of individuals 
(solutions). Each solution carries a set of genes. Within the context of SI 
tuning, a set carries all 6 PARAMETERS
%}
param_min = -1;
param_max = 1;
pop = (param_max - param_min).*rand(pop_size,param_size)+param_min;
fitness = zeros(pop_size,1);
parents = zeros(pop_size,2);
s = RandStream('mlfg6331_64');
iter=100;
for i = 1:iter
    
    %{
    Step 2 – Evaluation:
    After that, GA assess the fitness/quality of each solution using cost 
    functions. the error is then accumulated into the fitness value. The 
    solutions with best fitness value have higher chance to be selected for 
    mating process.
    %}
    for j = 1:pop_size
        err_sum = 0;
        for k = 1:dsize
            pred = 2*x(k,1) - x(k,2)+pop(j,1)*x(k,3)-...
            pop(j,2)*x(k,4)+pop(j,3)*x(k,5)+pop(j,4)*x(k,6);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        fitness(j,1) = err_sum;
    end
    %{
    Step 3 – Selection:
    Next, the breeding pairs from the population are chosen to generate new 
    offspring. 
    %}
    parents = roulette_wheel(fitness,pop_size);
    
    %{
    Step 4 – Crossover
    %}
    cross_prob = 0.4;
    new_pop = zeros(pop_size,param_size);
    for j = 1:pop_size
        for h = 1:param_size
            cross_poll = randsample(s,[0 1],1,true,[(1-cross_prob) cross_prob]);
            if cross_poll == 1
                new_pop(j,h) = pop(parents(j,2),h);
            else
                new_pop(j,h) = pop(parents(j,1),h);
            end
        end
    end
    pop = new_pop;
end
beta = pop(1,:);

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

%% Selection Functions
function parents = roulette_wheel(fitness,size)

    % Make probability
    max_ratio = 1.1;     % define upper limit to reverse the cost value
    max_lim = max(fitness)*max_ratio;
    fitness_reverse = max_lim - fitness;
    sum_val = sum(fitness_reverse);
    fitness_prob = fitness_reverse./sum_val;
    
    % Pick parents
    % https://uk.mathworks.com/help/stats/randsample.html
    parents = zeros(size,2);
    s = RandStream('mlfg6331_64');
    for i = 1:size
        index = randsample(s,size,1,true,fitness_prob);
%         parents(i,1) = fitness(index,1)
        parents(i,1) = index;
        index = randsample(s,size,1,true,fitness_prob);
%         parents(i,2) = fitness(index,1)
        parents(i,2) = index;
    end
end


