clc
clear

%% DATA
dsize = 100;

% Simple asymptotically linear data
x1 = linspace(1,20,dsize); x1 = x1';
x2 = randn(1)*linspace(1,20,dsize); x2 = x2';
x = [x1 x2];
% y = 5 + (0.96.^x(:,1)).*(x(:,1) + x(:,2));
y = 5 + 0.96.*x(:,1) + x(:,2);


%% Model Formulation
% y = a0 + a1*x1 + a2*x2;

%% Barnacles Mating Optimizer
% Barnacles Mating Optimizer: A new bio-inspired algorithm for solving
% engineering optimization problems
% Mohd Herwan Sulaiman a,?, Zuriani Mustaffa b, Mohd Mawardi Saari a, Hamdan Daniyal a
% https://doi.org/10.1016/j.engappai.2019.103330

pop_size = 100;         % size of population
param_size = 3;         % a0, a1, a2
iter=1000;               % number of iteration

% Init pop
param_min = -10;
param_max = 10;
pop_pos = (param_max - param_min).*rand(pop_size,param_size)+param_min;
new_pop_pos = zeros(pop_size,param_size);
fitness = zeros(pop_size,1);

% Algorithm parameters
pl = 20;    %  penis length

for i = 1:iter
    
    % Evaluate fitness
    for j = 1:pop_size
        err_sum = 0;
        for k = 1:dsize
            pred = pop_pos(j,1) + pop_pos(j,2)*x(k,1) + pop_pos(j,3)*x(k,2);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        fitness(j,1) = err_sum;
    end
    
    % sort
    [sorted,sorted_index] = sort(fitness);
    
    % Mating process
    new_pop_pos(1,:) = pop_pos(sorted_index(1,1),:);
    for j = 2:pl 
        p = (1 - 0).*rand(1)+0;
        new_pop_pos(j,:) = p.*new_pop_pos(1,:) + (1-p).*pop_pos(sorted_index(j,1),:);
    end
    
    for j = (pl+1):pop_size 
        p = (1 - 0).*rand(1)+0;
        new_pop_pos(j,:) = p.*pop_pos(sorted_index(j,1),:);
    end
    
    pop_pos = new_pop_pos;
end

% Plot with newly identified parameters
[sorted,sorted_index] = sort(fitness);
pred = pop_pos(sorted_index(1,1),1) + pop_pos(sorted_index(1,1),2).*x(:,1) + pop_pos(sorted_index(1,1),3).*x(:,2);

figure(1)
hold on
plot(y);
plot(pred);
hold off

%% Fitness Functions

function err = ise(pred,meas)
    err = (pred - meas)^2;
end