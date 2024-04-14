%%==============================================================================================================
%%  Least Squares B-Spline Approximation (LSBSA)
%  Authors: Alireza Amiri-Simkooei, Fatemeh Esmaeili, Roderik Lindenburgh, Delft University of Technology
%  Version: 1.0, April 2024
% ===============================================================================================================
%  This package consists of a series of functions used to implement least squares B-spline approximation (LSBSA)
%  LSBSA can be applied to 1D curve, 2D surface and 3D manifold fitting problems.
%  The codes were written by Alireza Amiri-Simkooei from Delft University of Technology in January 2022.
% ===============================================================================================================
function Function = lsbsa_package
Function.build_bspline = @build_bspline;
Function.A_matrix_1D = @A_matrix_1D;
Function.A_matrix_2D = @A_matrix_2D;
Function.A_matrix_3D = @A_matrix_3D;
Function.lsfit       = @lsfit;
end

function B = build_bspline(x,knots)
% This function compute the B-spline function values for the given values
% of x, and the given knots.
% INPUT:
%   x: vector of scattered positions for the observed values
%   knots: vector of places for the knots at which the pieces meet
% OUTPUT:
%   B: vector of Bspline functions value at x positions
m = length(x);
IDX = [1:m];
c = bspline(knots);
C = c.coefs;
n = length(C);
for i=1:n
    dx = x-knots(i);
    b = 0;
    for j=1:n
        b = b + C(i,j)*dx.^(n-j);
    end
    B(i,:) = b';
end
B0 = [];
IDX_N = [];
for i=1:n
    if i<n
        idx = find(x>=knots(i) & x<knots(i+1));
    else
        idx = find(x>=knots(i) & x<=knots(i+1));
    end
    B0 = [B0 B(i,idx)];
    IDX_N = [IDX_N IDX(idx)];
end
[II IDX_S] = sort(IDX_N);
B = B0(IDX_S)';
if length(B)~=length(x)
    error('Dimension problem')
end
end

function A = A_matrix_1D(x,knotx,degx)
% The function constructs the design matrix A of 1D LSBSA used for cutrve
% fitting problem.
% INPUT: x, knotx, degx
%   x: vector of scattered positions for the observed values
%   knotx: vector of positions for the knots at which the pieces meet
%   x values should always fall within the intervals defined by the knotx values,
%      so (no values of x should lie outside the knotx intervals)
%   degx: B-spline order: 1 (linear), 2 (quadratic), 3 (cubic), etc.
% OUTPUT:
%   A: the design matrix of 1D LSBSA
knotx0 = knotx;
mkx = mean(diff(knotx));
lkx = length(knotx);
m = length(x);
index_o = [1:m];
n = (lkx+degx-1);
A = zeros(m,n);
for i=1:degx
    knotx=[knotx(1)-mkx knotx knotx(end)+mkx];
end
nx = (lkx-1)+degx;
IX = [];
RowI = [0 0];
col = 0;
for i=1:lkx-1
    col = col+1;
    if i<lkx-1
        idx = find(x>=knotx0(i) & x<knotx0(i+1));
    else
        idx = find(x>=knotx0(i) & x<=knotx0(i+1));
    end
    sx = x(idx);
    ix = index_o(idx);
    IX = [IX ix];
    RowI = [RowI(1,2)+1 RowI(1,2)+length(idx)];
    colnew = col;
    for Kx=1:degx+1;
        Bx = build_bspline(sx,[knotx(i+Kx-1:i+Kx+degx)]);
        A(RowI(1,1):RowI(1,2), colnew) = Bx;
        colnew = colnew+1;
    end
end
[IX idxs] = sort(IX);
A = A(idxs,:);
end

function A = A_matrix_2D(x,y,knotx,knoty,degx,degy)
% The function constructs the design matrix A of 2D LSBSA used for surface
% fitting problem.
% INPUT: x, y, knotx, knoty, degx, degy
%   x and y: vector of scattered positions for the observed values in the plane x-y
%   knotx and knoty: vectors of positions for the knots at which the pieces meet
%   x and y values should always fall within the intervals defined by the knotx and knoty values,
%      so for example no values of x should lie outside the knotx intervals.
%   degx and degy: B-spline orders in x and y directions: 1 (linear), 2 (quadratic), 3 (cubic), etc.
% OUTPUT:
%   A: the design matrix of 2D LSBSA
knotx0 = knotx;
knoty0 = knoty;
mkx = mean(diff(knotx));
mky = mean(diff(knoty));
lkx = length(knotx);
lky = length(knoty);
nx = (lkx-1)+degx;
ny = (lky-1)+degy;
m = length(x);
n = nx*ny;
index_o = [1:m];
A = zeros(m,n);
for i = 1:degx
    knotx = [knotx(1)-mkx knotx knotx(end)+mkx];
end
for i = 1:degy
    knoty = [knoty(1)-mky knoty knoty(end)+mky];
end
count = 1;
for i = 1:degx+1
    for j = 1:degy+1
        ColIndex(i,j) = count;
        count = count+1;
    end
end
for i = 1:degx+1
    for j = degy+2:ny
        ColIndex(i,j) = count;
        count = count+1;
    end
end
for i = degx+2:nx
    for j = 1:ny
        ColIndex(i,j) = count;
        count = count+1;
    end
end
IX = [];
RowI = [0 0];
for i = 1:lkx-1
    if i<lkx-1
        idxx = find(x>=knotx0(i) & x<knotx0(i+1));
    else
        idxx = find(x>=knotx0(i) & x<=knotx0(i+1));
    end
    for j = 1:lky-1
        if j<lky-1
            idxy = find(y>=knoty0(j) & y<knoty0(j+1));
        else
            idxy = find(y>=knoty0(j) & y<=knoty0(j+1));
        end
        idx = intersect(idxx,idxy);
        sx = x(idx);
        sy = y(idx);
        ix = index_o(idx);
        IX = [IX ix];
        RowI = [RowI(1,2)+1 RowI(1,2)+length(idx)];
        ColJ = ColIndex(i:i+degx,j:j+degy);
        for Kx = 1:degx+1
            Bx = build_bspline(sx,[knotx(i+Kx-1:i+Kx+degx)]);
            for Ky = 1:degy+1
                By = build_bspline(sy,[knoty(j+Ky-1:j+Ky+degy)]);
                Ac = Bx.*By;
                ColJ0 = ColJ(Kx,Ky);
                A(RowI(1,1):RowI(1,2), ColJ0) = Ac;
            end
        end
    end
end
[IX idxs] = sort(IX);
A = A(idxs,:);
end

function A = A_matrix_3D(x,y,t,knotx,knoty,knott,degx,degy,degt)
% The function constructs the design matrix A of 2D LSBSA used for manifold
% fitting problem.
% INPUT: x, y, t, knotx, knoty, knott, degx, degy, degt
%   x, y and t: vectors of scattered positions for the observed values in the x-y-t space
%   knotx, knoty and knott: vectors of positions for the knots at which the pieces meet.
%   x, y and t values should always fall within the intervals defined by the knotx, knoty and knott values,
%      so for example no values of x should lie outside the knotx intervals.
%   degx, degy and degt: B-spline orders in x, y and t directions: 1 (linear), 2 (quadratic), 3 (cubic), etc.
% OUTPUT:
%   A: the design matrix of 3D LSBSA
knotx0 = knotx;
knoty0 = knoty;
knott0 = knott;
mkx = mean(diff(knotx));
mky = mean(diff(knoty));
mkt = mean(diff(knott));
lkx = length(knotx);
lky = length(knoty);
lkt = length(knott);
nx = (lkx-1)+degx;
ny = (lky-1)+degy;
nt = (lkt-1)+degt;
m = length(x);
n = nx*ny*nt;
index_o = [1:m];
A = zeros(m,n);
for i = 1:degx
    knotx = [knotx(1)-mkx knotx knotx(end)+mkx];
end
for i = 1:degy
    knoty = [knoty(1)-mky knoty knoty(end)+mky];
end
for i = 1:degt
    knott = [knott(1)-mkt knott knott(end)+mkt];
end
count = 1;
for i = 1:degx+1
    for j = 1:degy+1
        for k = 1:degt+1
            ColIndex(i,j,k) = count;
            count = count+1;
        end
    end
end
for i = 1:degx+1
    for j = 1:degy+1
        for k = degt+2:nt
            ColIndex(i,j,k) = count;
            count = count+1;
        end
    end
end
for i = 1:degx+1
    for j = degy+2:ny
        for k = 1:nt
            ColIndex(i,j,k) = count;
            count = count+1;
        end
    end
end
for i = degx+2:nx
    for j = 1:ny
        for k = 1:nt
            ColIndex(i,j,k) = count;
            count = count+1;
        end
    end
end
IX = [];
RowI = [0 0];
for i = 1:lkx-1
    if i<lkx-1
        idxx = find(x>=knotx0(i) & x<knotx0(i+1));
    else
        idxx = find(x>=knotx0(i) & x<=knotx0(i+1));
    end
    for j = 1:lky-1
        if j<lky-1
            idxy = find(y>=knoty0(j) & y<knoty0(j+1));
        else
            idxy = find(y>=knoty0(j) & y<=knoty0(j+1));
        end
        for k = 1:lkt-1
            if k<lkt-1
                idxt = find(t>=knott0(k) & t<knott0(k+1));
            else
                idxt = find(t>=knott0(k) & t<=knott0(k+1));
            end
            idx_xy = intersect(idxx,idxy);
            idx = intersect(idx_xy,idxt);
            sx = x(idx);
            sy = y(idx);
            st = t(idx);
            ix = index_o(idx);
            IX = [IX ix];
            RowI = [RowI(1,2)+1 RowI(1,2)+length(idx)];
            ColJ = ColIndex(i:i+degx,j:j+degy,k:k+degt);
            for Kx = 1:degx+1
                Bx = build_bspline(sx,[knotx(i+Kx-1:i+Kx+degx)]);
                for Ky = 1:degy+1
                    By = build_bspline(sy,[knoty(j+Ky-1:j+Ky+degy)]);
                    for Kt = 1:degt+1
                        Bt = build_bspline(st,[knott(k+Kt-1:k+Kt+degt)]);
                        Ac = Bx.*By.*Bt;
                        ColJ0 = ColJ(Kx,Ky,Kt);
                        A(RowI(1,1):RowI(1,2), ColJ0) = Ac;
                    end
                end
            end
        end
    end
end
[IX idxs] = sort(IX);
A = A(idxs,:);
end

function [xhat, yhat, ehat, sigma] = lsfit(A,y)
% The function applies the least squares method to estimate xhat, yhat, ehat and sigma
% INPUT: A: the m x n design matrix, y: observation vector of size m
% OUTPUT:
% xhat: least square estimate of x, yhat: least square estimate of y
% ehat: least square estimate of e, sigma: least square estimate of standard deviation of fit
[m n] = size(A);
xhat = inv(A'*A + 1e-8*eye(n))*A'*y; % the least squares estimate of coefficients (assuming Qy = I, an identity matrix)
yhat = A*xhat; % the least squares estimated observations
ehat = y - yhat; % the least squares estimates of the residuals
Var = ehat'*ehat/(m-n); % the least squares estimate of variance component (variance of fit + observations)
sigma = sqrt(Var);  % the estimated standard deviation of fit + observations
end
