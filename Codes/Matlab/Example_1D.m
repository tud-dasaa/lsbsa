%% Least Squares B-Spline Approximation (LSBSA)
%  Application of LSBSA to approximate 1D curve fitting problem
%  Authors: Alireza Amiri-Simkooei, Fatemeh Esmaeili, Roderik Lindenburgh, Delft University of Technology
%  Version: 1.0, April 2024
%%  
clear all
close all
%% Example on approximation of a known curve using LSBSA
%% load the LSBSA package as lsbsa
bs = lsbsa_package;
%% introduce m number of observations
u = [0:0.05:8*pi]'; % sampling u with interval of du=0.05 from 0 to 8*pi 
v = sin(u) + sin(u)./u; % a known function as: f(u)=sin(u)+sinc(u)
v(1) = 1;
m = length(v); %total number of observations
v = v+randn(m,1)/20; % add normally distributed noise to f(u)
y = v; % observations are also denoted as y
%% introduce knots intervals
cell = 2;
knotu = [min(u):cell:max(u) max(u)]; %knots u with intervals =2 (make sure all u's fall in these intervals)
figure(1);
%% LSBSA 1D approximation
for i=1:4
    degu = i; % B-spline order: 1:Linear, 2:Quadratic, 3:Cubic, 4: Quartic
    A = bs.A_matrix_1D(u,knotu,degu); % the design matrix based on Bspline cells and order
    [m n] = size(A);
    % least squares fit
    [xhat, yhat, ehat, sigma] = bs.lsfit(A,y)
    Std(i) = sigma;  % the estimated standard deviation of fit + observations
    up = [0:0.1:8*pi]' % u_p values with interval du=0.1 from 0 to 8*pi
    Ap = bs.A_matrix_1D(up,knotu,degu); % the design matrix for prediction y_p
    % predicted values for up
    yp = Ap*xhat;
    [int1, idx] = intersect(u,knotu);   
    % plot the results for  Linear, Quadratic, Cubic, Quartic orders
    subplot(2,2,i)
    plot(u,v,'.g','markersize',6)
    hold on
    plot(u,yhat,'-b','linewidth',1.5)
    plot(up,yp,'--','color',[1 0.7 0],'linewidth',1.5)
    plot(u(idx),yhat(idx),'ok','markersize',6)
    plot(u,ehat,'.','color',[1 0 0],'markersize',6)
    xlim([0 25.5])
    ylim([-1.5 2.])
    xlabel('u-axis')
    ylabel('v-axis')
    grid on
    box on 
    if i==1
        title('Linear spline')
    elseif i==2
        title('Quadratic spline')
    elseif i==3
        title('Cubic spline')
    else
        title('Quartic spline')
        legend('Data','Estimated y','Predicted y_p','Knots','Residuals')
    end
end
Std