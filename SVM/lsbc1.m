clc
clear

%% DATA
% Read Data
data = readmatrix('data1.xlsx');
x1 = data(:,1);
x2 = data(:,2);
x = [x1 x2];
y = data(:,3);

data_num = length(x1);
% Plot Data
hold on 
for i = 1:data_num
    if (y(i) == 1)
        plot(x1(i),x2(i),"rd");
    else
        plot(x1(i),x2(i),"ko");
    end
end
hold off
%% Create H (vector)
H = zeros(data_num);
for i = 1:data_num
    for j = 1:data_num
        H(i,j) = y(i)*y(j)*(dot(x(i,:),x(j,:)));
    end
end
%% Find alpha (vector)
% sym with for loop: https://www.mathworks.com/matlabcentral/answers/30074-for-loop-of-symbolic-variables-extracting-symbolic-coefficients
% alphasym = sym('alpha',[1 data_num]);
% eq = 0;
% for i = 1:data_num
%      eq = eq + alphasym(i)*y(i);
% %      assume(alphasym(i) >= 0);
% %      assume(alphasym(i) < 2);
% end
% alpha = solve(eq,alphasym)

% https://www.mathworks.com/help/optim/ug/optim.problemdef.optimizationproblem.solve.html
% https://www.mathworks.com/help/optim/ug/optimvar.html
% https://www.mathworks.com/help/optim/ug/convert-for-loop-constraints-static-analysis.html

alpha = optimvar('alpha',data_num);
prob = optimproblem('ObjectiveSense','max');
prob.Objective = 0;
cons1 = 0;
for i = 1:data_num
    prob.Objective = prob.Objective + alpha(i);
    cons1 = cons1 + alpha(i)*y(i);
end
prob.Objective = prob.Objective -(1/2)*alpha'*H*alpha;
prob.Constraints.cons1 = cons1 == 0;
prob.Constraints.cons2 = alpha >=0 ;
show(prob)
sol = solve(prob);
sol.alpha;

%% Find Weight (Vector)
w = zeros(1,2);
for i = 1:data_num
    w = w + (sol.alpha(i)*y(i))*x(i,:);
end

%% Find SVM indices
indices = zeros(1,1);
for i = 1:data_num
    if sol.alpha(i) > 0.0001
        indices = [indices; i];
    end
end
indices = indices(2:end);

%% Find b
b = 0;
for s = 1:length(indices)
    b = b + y(indices(s));
    for m = 1:length(indices)
        b = b - sol.alpha(indices(m))*y(indices(m))*(dot(x(indices(m),:),x(indices(s),:)));
    end
end
b = b/length(indices);

%% New point
x_new1 = [1 9];
y_new1 = sign(w*x_new1'+b)
x_new2 = [9 3];
y_new2 = sign(w*x_new2'+b)
x_new3 = [9 7];
y_new3 = sign(w*x_new3'+b)
x_new4 = [9 9];
y_new4 = sign(w*x_new4'+b)
x_new5 = [5 5];
y_new5 = sign(w*x_new5'+b)
x_new6 = [5 2];
y_new6 = sign(w*x_new6'+b)