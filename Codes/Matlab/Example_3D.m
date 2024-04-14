%% Least Squares B-Spline Approximation (LSBSA)
%  Application of LSBSA to approximate 3D manifold fitting problem
%  Authors: Alireza Amiri-Simkooei, Fatemeh Esmaeili, Roderik Lindenburgh, Delft University of Technology
%  Version: 1.0, April 2024
%% 
clear all
close all
%%  Approximation of a known 3D mathematical manifold z=f(u,v,t)
%% load the LSBSA package as lsbsa
bs = lsbsa_package;
%% introduce m number of observations randomly scattered
% generate a simulated data set of m=10000 points (-2<=u<=2, -2<=v<=2, 0<=t<=4)
m1 = 10000;
u = 4*(rand(m1,1)-0.5);
v = 4*(rand(m1,1)-0.5);
t = 4*(rand(m1,1)-0.0);
z = (1+t/20).*u.*exp(-u.^2 - v.^2) + v.*u.*(t/50); % known 3D mathematical function f(u,v,t)
% observations y (without error) in the model y = Ax
y = z;
%% introduce knot intervals for u, v and t
knotu = [-2:0.5:2]; % along u axis
knotv = [-2:0.5:2]; % along v axis
knott = [0:0.5:4];  % along t(time) axis
%% LSBSA 3D approximation (here using cubic spline)
degu = 3; % Bspline order along u axis: 1:Linear 2:Quadratic 3:Cubic 4: Quartic, etc
degv = 3; % Bspline order along v axis: 1:Linear 2:Quadratic 3:Cubic 4: Quartic, etc
degt = 3; % Bspline order along time: 1:Linear 2:Quadratic 3:Cubic 4: Quartic, etc
%% Apply the least squares method to estimate the unknown coefficients
A = bs.A_matrix_3D(u,v,t,knotu,knotv,knott,degu,degv,degt); % design matrix based on Bspline cells and order
[m n] = size(A);
% least squares fit
[xhat, yhat, ehat, sigma] = bs.lsfit(A,y)
Std = sigma;  % the estimated standard deviation of fit + observations
% plot original dataset
figure
scatter3(u,v,t,4,z,'filled')
colormap jet
colorbar
xlabel('u-axis')
ylabel('v-axis')
zlabel('t-axis')
grid on
box on
% plot least square results
figure
plot(ehat,'o')
xlabel('Observation ID')
ylabel('LS residual')
grid on
box on

%% predict the function values f(u,v,t) on a regular grid for a given t (time)
[U,V] = meshgrid(-2:0.1:2, -2:0.1:2); % grid size:0.1
[m1 n1] = size(U);
% predict the function values f(u,v,t=0,1,2,3) on a regular grid at time instances
t = [0 1 2 3];
for i=1:length(t)
    T = ones(m1,n1)*t(i);
    % design matrix for the grid points at time instances
    Ap = bs.A_matrix_3D(U(:),V(:),T(:),knotu,knotv,knott,degu,degv,degt);
    yp = Ap*xhat; % vector of predicted values of grid points at time (t(i))
    zp = yp;
    Zp = reshape(zp,m1,n1);
    % plot approximated surface at time t(i)
    figure
    surf(U,V,Zp)
    zlim([-0.6 0.6])
    colormap jet
    colorbar
    xlabel('u-axis')
    ylabel('v-axis')
    zlabel('Predicted f(u,v,t)')
    box on, grid on
    title(['t = ' num2str(t(i))])
    shg
end
