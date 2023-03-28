clc
clear

%% DATA
dsize = 100;

% Simple asymptotically linear data
x1 = linspace(1,20,dsize); x1 = x1';
x2 = randn(1)*linspace(1,20,dsize); x2 = x2';
x = [x1 x2];
y = 5 + (0.96.^x(:,1)).*(x(:,1) + x(:,2));

%% Model Formulation
% y = a0 + a1*x1 + a2*x2;

%% Grey Wolf Algorithm
% Seyedali Mirjalili a,?, Seyed Mohammad Mirjalili b, Andrew Lewis a
% Grey Wolf Optimizer - Advances in Engineering Software
% http://dx.doi.org/10.1016/j.advengsoft.2013.12.007

pop_size = 1000;
param_size = 3; % a0, a1, a2
iter=100000;

% Init GW pop
pos_min = -10;
pos_max = 10;
pop_pos = (pos_max - pos_min).*rand(pop_size,param_size)+pos_min;
fitness = zeros(pop_size,1);

a = 2;  % a linearly decreases from 2 to 0
a_decrement = a/iter;

% Limit of r1, r2
r_min = 0;
r_max = 1;

for i = 1:iter
    
    % Calculate fitness of each population
    for j = 1:pop_size
        err_sum = 0;
        for k = 1:dsize
            pred = pop_pos(j,1) + pop_pos(j,2)*x(k,1) + pop_pos(j,3)*x(k,2);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        fitness(j,1) = err_sum;
    end
    
    % Find alpha, beta, gamma (they may change in each iteration)
    [sorted,sorted_index] = sort(fitness);
    a_index = sorted_index(1);
    b_index = sorted_index(2);
    g_index= sorted_index(3);
    % Again, we don't know where the prey is, so we will assume it is near
    % all three alpha, beta, gamma. And we will update the other wolfs
    % around alpha, beta, gamma.
    
    % Update the remaining population (deltas)
    for j = 4:pop_size % start from 4 since the first 3 are alpha, beta, gamma
        
        % Find X1
        A1 = (2+a).*((r_max - r_min).*rand(1,param_size)+r_min)-a;  % A = 2a*r1-a
        C1 = 2.*((r_max - r_min).*rand(1,param_size)+r_min);        % C = 2*r2
        % where r1, r2 are random vector from 0 to 1
        
        D1 = abs(C1.*pop_pos(a_index,:)-pop_pos(j,:));
        X1 = pop_pos(a_index,:) - A1.*D1;
        
        % Find X2
        A2 = (2+a).*((r_max - r_min).*rand(1,param_size)+r_min)-a;  % A = 2a*r1-a
        C2 = 2.*((r_max - r_min).*rand(1,param_size)+r_min);        % C = 2*r2
        % where r1, r2 are random vector from 0 to 1
        
        D2 = abs(C2.*pop_pos(b_index,:)-pop_pos(j,:));
        X2 = pop_pos(b_index,:) - A2.*D2;
        
        % Find X3
        A3 = (2+a).*((r_max - r_min).*rand(1,param_size)+r_min)-a;  % A = 2a*r1-a
        C3 = 2.*((r_max - r_min).*rand(1,param_size)+r_min);        % C = 2*r2
        % where r1, r2 are random vector from 0 to 1
        
        D3 = abs(C3.*pop_pos(g_index,:)-pop_pos(j,:));
        X3 = pop_pos(g_index,:) - A3.*D3;
        
        % Update wolf position X = (X1 + X2 + X3)/3
        pop_pos(j,:) = (X1 + X2 + X3)./3;
    end
    
    % linearly decrease a
    a = a - a_decrement;
    
end

% Plot with newly identified parameters
pred = pop_pos(a_index,1) + pop_pos(a_index,2).*x(:,1) + pop_pos(a_index,3).*x(:,2);

figure(1)
hold on
plot(y);
plot(pred);
hold off
%% Fitness Functions

function err = ise(pred,meas)
    err = (pred - meas)^2;
end
