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

%% Artificial bee colony algorithm
% Dr. Dervis Karaboga, Erciyes University, Computer Engineering Department, Turkey
% http://www.scholarpedia.org/article/Artificial_bee_colony_algori0thm?ref=https://githubhelp.com

pop_size = 100;         % size of population
param_size = 3;         % a0, a1, a2
iter=100;               % number of iteration

% Init ABC pop
param_min = -10;
param_max = 10;
xm = (param_max - param_min).*rand(pop_size,param_size)+param_min;
fitness = zeros(pop_size,1);


% Define population of each type of bee
employed_ratio = 0.2;
onlooker_ratio = 0.4;
% scout_ratio = 1.0 - (employed_ratio + onlooker_ratio);

employed_size = round(employed_ratio*pop_size);
onlooker_size = round(onlooker_ratio*pop_size);
scout_size = pop_size - (employed_size + onlooker_size);

employed_index = zeros(employed_size,1); % store indices of employed bees
onlooker_index = zeros(onlooker_size,1); % store indices of onlooker bees
scout_index = zeros(scout_size,1);       % store indices of scout bees

vm = zeros(employed_size,param_size);   % store updated position of employed 
                                        % bees, used for onlooker bees
fitness_vm = zeros(employed_size,1);    % store fitness of updated employed
                                        % bee, used copuled with vm

% theta
% used for vmi = xmi + theta*(xmi-xki);
theta_min = -1;     
theta_max = 1;

for i = 1:iter
    
    % Calculate fitness of all solutions
    for j = 1:pop_size
        err_sum = 0;
        for k = 1:dsize
            pred = xm(j,1) + xm(j,2)*x(k,1) + xm(j,3)*x(k,2);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        fitness(j,1) = err_sum;
    end
    
    % Categorize the population
    [sorted,sorted_index] = sort(fitness);
    
    employed_index = sorted_index(1:employed_size,1);
    onlooker_index = sorted_index((employed_size+1):(employed_size+onlooker_size),1);
    scout_index = sorted_index((employed_size+onlooker_size+1):end,1);
    
    % temporary store the current population because we will update 
    % directly on xm, but still need previous xm for picking xki
    xm_temp = xm;
    
    % Update employed bees
    for j = 1:employed_size
        
        % choose a random theta in range
        theta = (theta_max - theta_min).*rand(1)+theta_min;
        xmi = xm(employed_index(j),:); 
        % choose random index of xki
        rand_index = randi(pop_size);
        vmi = xmi + theta.*(xmi-xm_temp(rand_index,:));
        
        % Find fitness of vmi
        err_sum = 0;
        for k = 1:dsize
            pred = vmi(1,1) + vmi(1,2)*x(k,1) + vmi(1,3)*x(k,2);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        fitness_vmi = err_sum;
        
        % greedy between xmi and vmi
        if fitness_vmi < fitness(employed_index(j),1)
            xm(employed_index(j),:) = vmi;
            fitness(employed_index(j),1) = fitness_vmi;
        end
        
        % assign to vm for usage of onlooker bees
        vm(j,:) = xm(employed_index(j),:);
        fitness_vm(j,1) = fitness(employed_index(j),1);
    end
    
    % Update onlooker bees
    
    % Make probability of vm
    max_ratio = 1.1;     % define upper limit to reverse the cost value
    max_lim = max(fitness_vm)*max_ratio;
    fitness_reverse = max_lim - fitness_vm;
    sum_val = sum(fitness_reverse);
    fitness_prob = fitness_reverse./sum_val;
    
    s = RandStream('mlfg6331_64');
    for j = 1:onlooker_size
        
        % take a random index in vm, based on calculated prob
        index = randsample(s,employed_size,1,true,fitness_prob);
        
        % Find fitness of randomly chosen vmi
        err_sum = 0;
        for k = 1:dsize
            pred = vm(index,1) + vm(index,2)*x(k,1) + vm(index,3)*x(k,2);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        fitness_vmi = err_sum;
        
        % greedy between xmi and vmi
        if fitness_vmi < fitness(onlooker_index(j),1)
            xm(onlooker_index(j),:) = vmi;
            fitness(onlooker_index(j),1) = fitness_vmi;
        end
        
    end
    
    % Update scout bees
    
    for j = 1:scout_size
        xm(scout_index(j),:) = (param_max - param_min).*rand(1,param_size)+param_min;
    end
    
end

% Plot with newly identified parameters
[sorted,sorted_index] = sort(fitness);
pred = xm(sorted_index(1,1),1) + xm(sorted_index(1,1),2).*x(:,1) + xm(sorted_index(1,1),3).*x(:,2);

figure(1)
hold on
plot(y);
plot(pred);
hold off

%% Fitness Functions

function err = ise(pred,meas)
    err = (pred - meas)^2;
end




