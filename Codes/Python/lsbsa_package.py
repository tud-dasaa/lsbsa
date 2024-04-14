# =============================================================================================================
# Least Squares B-Spline Approximation (LSBSA)
# Authors: Alireza Amiri-Simkooei, Fatemeh Esmaeili, Roderik Lindenburgh, Delft University of Technology
# Version: 1.0, April 2024
# =============================================================================================================
# This package consists of a series of functions used to implement least squares B-spline approximation (LSBSA)
# LSBSA can be applied to 1D curve, 2D surface and 3D manifold fitting problems.
# The codes were written in Matlab and Python by Alireza Amiri-Simkooei in January 2022.
# Parts of the Matlab codes were converted to Python by Stephen Goldebeld in November 2022.
# =============================================================================================================
# Load the necessary packages 
import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline, PPoly
def build_bspline(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Build bspline B matrix for 1D.
    Input
    ----------------------------------
    x: np.ndarray of data points.
    knots: np.ndarray Knots.
    Output
    ----------------------------------
    B: np.ndarray 1D bspline B matrix.
    """
    m = len(x)
    k = len(knots)
    c = BSpline.basis_element(knots)
    C = PPoly.from_spline(c.tck).c.T[k-2:-k+2]
    n = C.shape[0]
    B = np.zeros((n, m))
    for i in range(n):
        dx = (x - knots[i])
        b = 0
        for j in range(n):
            b += C[i, j] * dx ** ( n - j - 1 )
        B[i] = b.T
    B0 = []
    index = []
    for i in range(n):
        idx, *_ = np.where(
            ( x >= knots[i] ) &
            ( x <= knots[-1] if i == n - 1 else x < knots[i+1]))
        B0 = [*B0, *B[i, idx]]
        index.extend(idx)
    B = np.array(B0)[np.argsort(index)]
    B = B.reshape((m, 1))
    if max(B.shape) != max(B.shape):
        raise(Exception('Dimension problem'))
    return B

def A_matrix_1D(x, kx, degx):
    kx0 = kx
    kx_m = np.mean(np.diff(kx))
    kx_len = len(kx)
    m = len(x)
    index_o = np.arange(1, m+1)
    n = (kx_len + degx - 1)
    A = np.zeros((m, n))
    for i in range(1, degx + 1):
        kx = np.concatenate(([kx[0] - kx_m], kx, [kx[-1] + kx_m]))
    nx = (kx_len - 1) + degx
    RowI = [0, 0]
    ix = []
    col = 0
    for i in range(kx_len - 1):
        idx, *_ = np.where( ( x >= kx0[i]) & (x <= kx0[-1] if i == kx_len - 2 else x < kx0[i+1] ) )
        sx = x[idx]
        # Add them to the main indices
        ix.extend(idx)
        RowI = [RowI[1], RowI[1]+len(idx)]
        colnew = col
        for Kx in range(degx + 1):
            Bx = build_bspline(sx, kx[i + Kx : i + Kx + degx + 2])
            A[RowI[0]:RowI[1], colnew] = Bx.reshape(-1)
            colnew += 1
        col += 1
    return A[np.argsort(ix)]

def A_matrix_2D(
    x: np.ndarray,
    y: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    degx: int,
    degy: int) -> np.ndarray:
    """
    Build bspline A matrix for 2D.
    Parameters
    ----------
    x : np.ndarray
        X-coords.
    y : np.ndarray
        Y-coords.
    kx : np.ndarray
        X-knots.
    ky : np.ndarray
        Y-knots.
    degx : np.ndarray
        Degree in X.
    degy : np.ndarray
        Degree in Y.
    Returns
    -------
    A : np.ndarray
        2D bspline A matrix.
    """
    kx0 = kx
    ky0 = ky
    kx_m = np.mean(np.diff(kx))
    ky_m = np.mean(np.diff(ky))
    kx_len = max(kx.shape)
    ky_len = max(ky.shape)
    n_x = ( kx_len - 1 ) + degx
    n_y = ( ky_len - 1 ) + degy
    m = max(x.shape)
    n = n_x * n_y
    # Storage matrix for column indices
    col_i = np.zeros((n_x, n_y), dtype=int)
    A = np.zeros((m, n))
    for i in range(degx):
        kx = [kx[0] - kx_m, *kx, kx[-1] + kx_m]
    for j in range(degy):
        ky = [ky[0] - ky_m, *ky, ky[-1] + ky_m]
    count = 0
    for i in range(degx+1):
        for j in range(degy+1):
            col_i[i, j] = count
            count += 1
    for i in range(degx+1):
        for j in range(degy+1, n_y):
            col_i[i, j] = count
            count += 1
    for i in range(degx+1, n_x):
        for j in range(n_y):
            col_i[i, j] = count
            count += 1
    ix = []
    row_i = [0, 0]
    for i in range(kx_len - 1):
        idxx, *_ = np.where(
            ( x >= kx0[i] ) &
            ( x <= kx0[-1] if i == kx_len - 2 else x < kx0[i+1] ) )
        for j in range(ky_len - 1):
            idxy, *_ = np.where(
                ( y >= ky0[j] ) &
                ( y <= ky0[-1] if j == ky_len - 2 else y < ky0[j+1] ) )
            # Get common indices
            idx = np.intersect1d(idxx, idxy)
            # Select coordinates from these indices
            sx = x[idx]
            sy = y[idx]
            # Add them to the main indices
            ix.extend(idx)
            # Indices for matrix building
            row_i = [row_i[1], row_i[1] + len(idx)]
            col_j = col_i[i : i + degx + 1, j : j + degy + 1]
            for Kx in range(degx + 1):
                Bx = build_bspline(sx, kx[i + Kx : i + Kx + degx + 2])
                for Ky in range(degy + 1):
                    By = build_bspline(sy, ky[j + Ky : j + Ky + degy + 2])
                    Ac = Bx * By
                    col_j0 = col_j[Kx, Ky]
                    A[row_i[0] : row_i[1], col_j0] = Ac.reshape(-1)
    return A[np.argsort(ix)]


def A_matrix_3D(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kt: np.ndarray,
    degx: int,
    degy: int,
    degt: int) -> np.ndarray:
    """
    Build bspline A matrix for 3D.
    Parameters
    ----------
    x : np.ndarray
        X-coords.
    y : np.ndarray
        Y-coords.
    kx : np.ndarray
        X-knots.
    ky : np.ndarray
        Y-knots.
    degx : np.ndarray
        Degree in X.
    degy : np.ndarray
        Degree in Y.
    Returns
    -------
    A : np.ndarray
        3D bspline A matrix.
    """
    kx0 = kx
    ky0 = ky
    kt0 = kt
    kx_m = np.mean(np.diff(kx))
    ky_m = np.mean(np.diff(ky))
    kt_m = np.mean(np.diff(kt))
    kx_len = max(kx.shape)
    ky_len = max(ky.shape)
    kt_len = max(kt.shape)
    n_x = ( kx_len - 1 ) + degx
    n_y = ( ky_len - 1 ) + degy
    n_t = ( kt_len - 1 ) + degt
    m = max(x.shape)
    n = n_x * n_y * n_t
    # Storage matrix for column indices
    col_i = np.zeros((n_x, n_y, n_t), dtype=int)
    A = np.zeros((m, n))
    for i in range(degx):
        kx = [kx[0] - kx_m, *kx, kx[-1] + kx_m]
    for j in range(degy):
        ky = [ky[0] - ky_m, *ky, ky[-1] + ky_m]
    for k in range(degt):
        kt = [kt[0] - kt_m, *kt, kt[-1] + kt_m]
    count = 0
    for i in range(degx+1):
        for j in range(degy+1):
            for k in range(degt+1):
                col_i[i, j, k] = count
                count += 1
    for i in range(degx+1):
        for j in range(degy+1):
            for k in range(degt+1, n_t):
                col_i[i, j, k] = count
                count += 1
    for i in range(degx+1):
        for j in range(degy+1, n_y):
            for k in range(n_t):
                col_i[i, j, k] = count
                count += 1
    for i in range(degx+1, n_x):
        for j in range(n_y):
            for k in range(n_t):
                col_i[i, j, k] = count
                count += 1
    ix = []
    row_i = [0, 0]
    for i in range(kx_len - 1):
        idxx, *_ = np.where(
            ( x >= kx0[i] ) &
            ( x <= kx0[-1] if i == kx_len - 2 else x < kx0[i+1] ) )
        for j in range(ky_len - 1):
            idxy, *_ = np.where(
                ( y >= ky0[j] ) &
                ( y <= ky0[-1] if j == ky_len - 2 else y < ky0[j+1] ) )
            for k in range(kt_len - 1):
                idxt, *_ = np.where(
                    ( t >= kt0[k] ) &
                    ( t <= kt0[-1] if k == kt_len - 2 else t < kt0[k+1] ) )
                # Get common indices
                idx_xy = np.intersect1d(idxx, idxy)
                idx = np.intersect1d(idx_xy, idxt)
                # Select coordinates from these indices
                sx = x[idx]
                sy = y[idx]
                st = t[idx]
                # Add them to the main indices
                ix.extend(idx)
                #print(len(idx))
                # Indices for matrix building
                row_i = [row_i[1], row_i[1] + len(idx)]
                col_j = col_i[i:i+degx+1, j:j+degy+1, k:k+degt+1]
                for Kx in range(degx + 1):
                    Bx = build_bspline(sx, kx[i + Kx : i + Kx + degx + 2])
                    for Ky in range(degy + 1):
                        By = build_bspline(sy, ky[j + Ky : j + Ky + degy + 2])
                        for Kt in range(degt + 1):
                            Bt = build_bspline(st, kt[k + Kt : k + Kt + degt + 2])
                            Ac = Bx * By * Bt
                            col_j0 = col_j[Kx, Ky, Kt]
                            A[row_i[0] : row_i[1], col_j0] = Ac.reshape(-1)
    return A[np.argsort(ix)]

def lsfit(A, y):
    xhat = np.dot(linalg.inv(np.dot(A.T, A)), np.dot(A.T, y))
    yhat = np.dot(A, xhat)
    ehat = yhat - y
    m, n = A.shape
    sigma = np.sqrt(np.dot(ehat.T, ehat) / (m - n))
    return xhat, yhat, ehat, sigma