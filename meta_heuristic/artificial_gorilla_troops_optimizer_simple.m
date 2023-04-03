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

%% Artificial gorilla troops optimizer
% Benyamin Abdollahzadeh, Farhad Soleimanian Gharehchopogh, Seyedali Mirjalili
% Artificial gorilla troops optimizer: A new nature-inspired metaheuristic 
% algorithm for global optimization problems
% https://doi.org/10.1002/int.22535

pop_size = 100;         % size of population
param_size = 3;         % a0, a1, a2
iter=100;               % number of iteration

p = (1 - 0)*rand(1)+0;  % eq 1
W = 0.8;
beta = 3;

% Init pop
param_min = -10;
param_max = 10;
pop_pos = (param_max - param_min).*rand(pop_size,param_size)+param_min;
GX = zeros(pop_size,param_size);
fitness = zeros(pop_size,1);

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


% start training
for i = 1:iter
    % Update a, C using eq 2,4 
    r4 = (1 - 0)*rand(1)+0;
    F = cos(2*r4)+1;
    C = F*(1-(i/iter));         % eq 2
    l = (1 - (-1))*rand(1)+(-1);
    L = C*l;                    % eq 4
    
    % Go through whole pop again
    for j = 1:pop_size
        % Exploration phase
        % Update pop pos with eq 1
        
        randnum = (1 - 0)*rand(1)+0;
        if randnum < p
            r1 = (1 - 0)*rand(1,param_size)+0;
            GX(j,:) = (param_max - param_min).*r1+param_min;
        elseif randnum >= 0.5
            Z = (C - (-C))*rand(1)+(-C);    % eq 6
            H = Z.*pop_pos(j,:);            % eq 7
            r2 = (1 - 0)*rand(1)+0;
            GX(j,:) = (r2 - C).*pop_pos(j,:) + L.*H;    % eq 8
        else
            index = randi(pop_size);
            r3 = (1 - 0)*rand(1)+0;
            GX(j,:) = pop_pos(j,:) - L.*(L.*(pop_pos(j,:) - pop_pos(index,:)) + r3.*(pop_pos(j,:) - pop_pos(index,:)));
        end
    end
    
    % Find fitness again, if it is better then replace
    for j = 1:pop_size
        err_sum = 0;
        for k = 1:dsize
            pred = GX(j,1) + GX(j,2)*x(k,1) + GX(j,3)*x(k,2);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        
        if err_sum < fitness(j,1)
            fitness(j,1) = err_sum;
            pop_pos(j,:) = GX(j,:);
        end
       
    end
    
    % The best solution is silverback
    [fitness_silver, index_silver] = min(fitness);
    x_silver = pop_pos(index_silver,:);
    
    for j = 1:pop_size
        % Exploitation phase
        if C >= W
            % Yes -> Update pop pos with eq 7
            g = 2^L;        % eq 9
            sum = zeros(1,param_size);
            for h =1:pop_size
               index = randi(pop_size);
               sum = sum + pop_pos(index,:);
            end
            M = (abs( (1/pop_size).*sum  ).^g).^(1/g);
            GX(j,:) = L.*M.*(pop_pos(j,:) - x_silver) + pop_pos(j,:);
        else
            % No -> Update pop pos with eq 10
            randnum = (1 - 0)*rand(1)+0;
            if randnum >= 0.5
                E = (pop_size - (-pop_size))*rand(1)+(-pop_size);
            else
                E = (1 - 0)*rand(1)+0;
            end % eq 13
            A = beta*E; % eq 12
            r5 = (1 - 0)*rand(1)+0;
            Q = 2*r5 - 1; % eq 11
            GX(j,:) = x_silver - (x_silver.*Q-pop_pos(j,:).*Q).*A;
        end
    end
    
    % Find fitness again, if it is better then replace
    for j = 1:pop_size
        err_sum = 0;
        for k = 1:dsize
            pred = GX(j,1) + GX(j,2)*x(k,1) + GX(j,3)*x(k,2);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        
        if err_sum < fitness(j,1)
            fitness(j,1) = err_sum;
            pop_pos(j,:) = GX(j,:);
        end
       
    end
    % The best solution is silverback
    [fitness_silver, index_silver] = min(fitness);
    x_silver = pop_pos(index_silver,:);
    
end % end training

% Plot with newly identified parameters
pred = pop_pos(index_silver,1) + pop_pos(index_silver,2).*x(:,1) + pop_pos(index_silver,3).*x(:,2);

figure(1)
hold on
plot(y);
plot(pred);
hold off

%% Fitness Functions

function err = ise(pred,meas)
    err = (pred - meas)^2;
end