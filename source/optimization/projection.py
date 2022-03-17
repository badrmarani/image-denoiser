#!/usr/bin/env python3

# standard libraries
import time
import os

# third party
import numpy as np
from matplotlib.pyplot import *
from tqdm import tqdm
import cv2


# internal references
from source.utils.noise import gaussian
from source.utils.operators import div
from source.utils.operators import grad
from source.utils.operators import hessian
from source.utils.operators import hessian_adjoint

def norm_in_Rn (u) :
    return np.sqrt (sum (np.power (u,2)))


def projection_on_D (q) :
    a, b = np.shape (q) [0:2]
    D = q

    CRIT = []
    SNR = []
    PSNR = []

    x = []
    for i in range (a) :
        for j in range (b) :
            x = np.array([ q [0][i][j], q [1][i][j] ])

            if (y := norm_in_Rn (x)) > 1 :
                D [0][i][j] = D [0][i][j] / y
                D [1][i][j] = D [1][i][j] / y
    return D

def projection_on_D2 (q) :
    a, b = np.shape (q) [0:2]
    D=q

    for i in range(a) :
        for j in range(b) :
            x = np.array([ q[0][i][j] , q[1][i][j], q[2][i][j], q[3][i][j] ])

            if (y := norm_in_Rn (x)) >  1 :
                D[0][i][j] = D[0][i][j] / y
                D[1][i][j] = D[1][i][j] / y
                D[2][i][j] = D[2][i][j] / y
                D[3][i][j] = D[3][i][j] / y

    return D


def projection1 (image, mean = 0, sigma = 15, epsilon = 1, L = 10,
                    step_size = .20, max_iter = 100) :

    rho = 1/(33*(L)**2)

    (n,m) = np.shape(image)
    p = np.array ([np.zeros((n,m)), np.zeros((n,m))])

    CRIT = []

    new_p = projection_on_D (p + np.dot (2*rho*L, grad( L * div(p) - image)))

    u = image - L * div (p)
    u_new = image - L * div (new_p)

    for i in range (max_iter) :
        p = new_p
        new_p = projection_on_D ( new_p +  np.dot (2*rho*L, grad( L*div(new_p) - image )))

        u = image- L * div(p)
        u_new = image - L * div (new_p)

        if (x := np.max (np.abs (u_new - u))) < epsilon :
            pass

        CRIT.append (x)

    PSNR = [0]
    SNR = [0]
    return u_new, CRIT, SNR, PSNR

def projection2 (image, mean = 0, sigma = 15, epsilon = 1, L = 10,
                    step_size = .20, max_iter = 100) :

    print (L, sigma)

    tau = 1/(33*(L)**2)

    CRIT = []

    (n,m) = np.shape (image)

    q = np.array ([np.zeros((n,m)), np.zeros((n,m)), np.zeros((n,m)), np.zeros((n,m))])

    new_q = projection_on_D2 (q - np.dot (2*tau*L, hessian( L*hessian_adjoint(q) - image)))

    u = image - L * hessian_adjoint (q)
    u_new = image - L * hessian_adjoint (new_q)

    for i in range (max_iter) :
        q = new_q
        new_q = projection_on_D2 ( q - np.dot (2 *tau*L, hessian (L*hessian_adjoint(q) - image )))

        u = image - L * hessian_adjoint (q)
        u_new = image - L * hessian_adjoint (new_q)

        if (x := np.max (np.abs (u_new - u))) < epsilon :
            pass

        CRIT.append (x)

    PSNR = [0]
    SNR = [0]
    return u_new, CRIT, SNR, PSNR
