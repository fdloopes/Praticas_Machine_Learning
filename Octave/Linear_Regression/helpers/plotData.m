function plotData(x, y, xLabel, yLabel)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; % open a new figure window

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%

plot(x, y, 'rx', 'MarkerSize', 10);       % Plot the data    
ylabel(yLabel);                           % Set the axis y-axis label
xlabel(xLabel);                           % Set the axis x-axis label

% ============================================================

end
