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

%% African Vulture Optimization
% African vultures optimization algorithm: A new nature-inspired
% metaheuristic algorithm for global optimization problems
% Benyamin Abdollahzadeh a, Farhad Soleimanian Gharehchopogh a,*, 
% Seyedali Mirjalili b,c
% https://doi.org/10.1016/j.cie.2021.107408

pop_size = 1000;         % size of population
param_size = 3;         % a0, a1, a2
iter=100;               % number of iteration

% Init AVOA pop
param_min = -10;
param_max = 10;
pop_pos = (param_max - param_min).*rand(pop_size,param_size)+param_min;
fitness = zeros(pop_size,1);

% For random with probablity
s = RandStream('mlfg6331_64');

w = 2;                  % in eq (3)
P1 = 0.4;               % if-else con #1
P2 = 0.4;               % if-else con #2    
P3 = 0.4;               % if-else con #3
beta = 1.5;             % in eq (18) - LF (if-else con #3)

lb = param_min;         % in eq (8)
ub = param_max;         % in eq (8)

for i = 1:iter
    
    % Calculate fitness of all solutions
    for j = 1:pop_size
        err_sum = 0;
        for k = 1:dsize
            pred = pop_pos(j,1) + pop_pos(j,2)*x(k,1) + pop_pos(j,3)*x(k,2);
            err = ise(pred,y(k,1));
            err_sum = err_sum + err;
        end
        fitness(j,1) = err_sum;
    end
    
    % Find best and second best
    [sorted,sorted_index] = sort(fitness);
    pb1 = 1 - sorted(1,1)/(sorted(1,1) + sorted(2,1));
    pb2 = 1 - pb1;
    
    % go through the rest of population except two best
    for k = 3:pop_size
        j = sorted_index(k,1);    % take index of the rest
        
        % Roulette wheel between two best
        index = randsample(s,2,1,true,[pb1 pb2]);
        if index == 1
            Ri = pop_pos(sorted_index(1,1),:);
        else
            Ri = pop_pos(sorted_index(2,1),:);
        end
        
        % Eq (3) & (4)
        h = (2 - (-2))*rand(1)+(-2);
        t = h*( (sin( (pi/2)*(i/iter) ))^w + cos( (pi/2)*(i/iter) ) -1); % eq (3)
        rand1 = (1 - 0)*rand(1)+0;
        z = (1 - (-1))*rand(1)+(-1);
        F = (2*rand1 + 1)*z*(1 - i/iter) + t; % eq (4)      
        
        if F >= 1
            % Yes -> Exploration
            randp1 = (1 - 0)*rand(1)+0;
            if P1 >= randp1
                % yes -> use eq (6)
                randx =  (1 - 0)*rand(1)+0;
                X = 2*randx;
                Di = abs(X.*Ri - pop_pos(j,:));
                pop_pos(j,:) = Ri - Di.*F; % update vulture pos
            else
                % no -> use eq (8)
                rand2 = (1 - 0)*rand(1)+0;
                rand3 = (1 - 0)*rand(1)+0;
                pop_pos(j,:) = Ri - F + rand2*((ub-lb)*rand3 + lb); % update vulture pos
            end % randp1
        else
            % No -> Exploitation
            if F >= 0.5
                % Yes -> randp2
                randp2 = (1 - 0)*rand(1)+0;
                if P2 >= randp2
                    % Yes -> use eq (10)
                    randx =  (1 - 0)*rand(1)+0;
                    X = 2*randx;
                    Di = abs(X.*Ri - pop_pos(j,:));
                    dt = Ri - pop_pos(j,:);
                    rand4 = (1 - 0)*rand(1)+0;
                    pop_pos(j,:) = Di.*(F+rand4) - dt; % update vulture pos
                else
                    % No -> use eq (13)
                    rand5 = (1 - 0)*rand(1)+0;
                    rand6 = (1 - 0)*rand(1)+0;
                    S1 = Ri.*(rand5.*pop_pos(j,:)./(2*pi)).*cos(pop_pos(j,:));
                    S2 = Ri.*(rand6.*pop_pos(j,:)./(2*pi)).*sin(pop_pos(j,:));
                    pop_pos(j,:) = Ri - (S1 + S2); % update vulture pos
                end
            else
                % No -> randp3
                randp3 = (1 - 0)*rand(1)+0;
                if P3 >= randp3
                    % Yes -> eq (16)
                    best1 = pop_pos(sorted_index(1,1),:);
                    best2 = pop_pos(sorted_index(2,1),:);
                    A1 = best1 - (best1.*pop_pos(j,:))/(best1 - pop_pos(j,:).^2).*F;
                    A2 = best2 - (best2.*pop_pos(j,:))/(best2 - pop_pos(j,:).^2).*F;
                    pop_pos(j,:) = (A1+A2)/2;   % update vulture pos
                else
                    % No -> eq (17)
                    dt = Ri - pop_pos(j,:);
                    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma(1+beta^2)*beta*2*(beta-1)/2))^(1/beta);
                    u = (1 - 0)*rand(1)+0;
                    v = (1 - 0)*rand(1)+0;
                    LF = 0.01*(u*sigma)/abs(v)&(1/beta);
                    pop_pos(j,:) = Ri - abs(dt).*(LF*F); % update vulture pos
                end % randp3
            end % randp2
        end % F
    end % k = 3:pop_size
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
