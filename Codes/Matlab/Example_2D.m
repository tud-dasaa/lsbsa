%% Least Squares B-Spline Approximation (LSBSA)
%  Application of LSBSA to approximate 2D surface fitting problem
%  Authors: Alireza Amiri-Simkooei, Fatemeh Esmaeili, Roderik Lindenburgh, Delft University of Technology
%  Version: 1.0, April 2024
%% 
clear all
close all
%% Example on approximation of a known surface z=f(u,v) using LSBSA
%% load the LSBSA package as lsbsa
bs = lsbsa_package;
%% introduce  m number of observations
% generate 20000 random positions in u-v plane between -2 and 2
m1 = 20000;
u = 4*(rand(m1,1)-0.5);
v = 4*(rand(m1,1)-0.5);
z = u.*exp(-u.^2 - v.^2); % compute the known function f(u,v)=u*exp(-u^2 - v^2) on these points
% observations y (without error) in the model y = Ax
y = z;
%% introduce knot intervals for u and v 
knotu = [-2:0.4:2]; % along u axis
knotv = [-2:0.4:2]; % along v axis
%% LSBSA 2D approximation
%for it = 1:4
for it = 3:3 % considering here the cubic b-spline
    degu = it; % Bspline order along u axis : 1:Linear 2:Quadratic 3:Cubic 4: Quartic, etc
    degv = it; % Bspline order along v axis : 1:Linear 2:Quadratic 3:Cubic 4: Quartic, etc
    A = bs.A_matrix_2D(u,v,knotu,knotv,degu,degv); % design matrix based on Bspline cells and order
    [m n] = size(A);
    % least squares fit
    [xhat, yhat, ehat, sigma] = bs.lsfit(A,y)
    Std(it) = sigma;  % the estimated standard deviation of fit + observations
    % plot least square results
    figure
    scatter(u,v,5,ehat,'filled')
    colormap jet
    colorbar
    xlabel('u-axis')
    ylabel('v-axis')
    box on, grid on
    % predict the function values f(u,v) on a regular grid 
    [U,V] = meshgrid(-2:0.05:2, -2:0.05:2); % grid size:0.1
    Z = U .* exp(-U.^2 - V.^2);
    [m1 n1] = size(U);
    up = U(:);
    vp = V(:);
    Ap = bs.A_matrix_2D(up,vp,knotu,knotv,degu,degv);% design matrix for the grid points
    zp = Ap*xhat; %vector of predicted values of grid points
    Zp = reshape(zp,m1,n1);
    % plot true surface
    figure
    surf(U,V,Z)
    colormap jet
    colorbar
    xlabel('u-axis')
    ylabel('v-axis')
    zlabel('True f(u,v)')
    box on, grid on
    % plot approximated surface
    figure
    surf(U,V,Zp)
    colormap jet
    colorbar
    xlabel('u-axis')
    ylabel('v-axis')
    zlabel('Approximated f(u,v)')
    box on, grid on
    % plot approximation error
    figure
    surf(U,V,Z-Zp)
    colormap jet
    colorbar
    xlabel('u-axis')
    ylabel('v-axis')
    zlabel('Approximation error')
    box on, grid on
    view([0 90])
end
Std(it)
