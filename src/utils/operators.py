#!/usr/bin/env python3

# third party
import numpy as np

def grad (P) :
    a, b = np.shape (P)[0:2]

    Px = np.zeros ((a,b))
    Py = np.zeros ((a,b))

    for i in range (a) :
        for j in range (b) :
            if i < a-1 :
                Px [i,j] = P [i+1,j] - P [i,j]
            elif i == a-1 :
                Px[i,j] = 0

            if j < b-1 :
                Py [i,j] = P [i,j+1] - P [i,j]
            elif j == b-1 :
                Py [i,j] = 0

    return Px, Py

def div (P) :
    a, b = np.shape (P)[1:3]

    Px = np.zeros ((a,b))
    Py = np.zeros ((a,b))

    for i in range (a) :
        for j in range (b) :
            if 0 < i < a-1 :
                Px [i,j] = P [0,i,j] - P [0,i-1,j]
            elif i == 0 :
                Px [i,j] = P [0,i,j]
            elif i == a-1 :
                Px [i,j] = - P [0,i-1,j]

            if 0 < j < a-1 :
                Py [i,j] = P [1,i,j] - P [1,i,j-1]
            elif j == 0 :
                Py [i,j] = P [1,i,j]
            elif j == b-1 :
                Py [i,j] = - P [1,i,j-1]

    return Px + Py

def hessian (H) :
    a, b = np.shape (H)

    H11 = np.zeros ((a,b))
    H12 = np.zeros ((a,b))
    H21 = np.zeros ((a,b))
    H22 = np.zeros ((a,b))

    for i in range (a) :
        for j in range (b) :
            if 0 < i < a-1 :
                H11 [i,j] = H[i+1,j] - 2 * H[i,j] + H[i-1,j]
            elif i == 0 :
                H11 [i,j] = H[i+1,j] - H[i,j]
            elif i == a-1 :
                H11 [i,j] = H[i-1,j] - H[i,j]

            if 0 < i <= a-1 and 0 <= j < b-1 :
                H12 [i,j] = H[i,j+1] - H[i,j] - H[i-1,j+1] + H[i-1,j]
            elif i == 0 or i == a-1 :
                H12 [i,j] = 0

            if 0 <= i < a-1 and 0 < j <= b-1 :
                H21 [i,j] = H[i+1,j] - H[i,j] - H[i+1,j-1] + H[i,j-1]
            elif i == 0 or i == a-1 :
                H21 [i,j] = 0

            if 0 < j < b-1 :
                H22 [i,j] = H[i,j+1] - 2 * H[i,j] + H[i,j-1]
            elif j == 0 :
                H22 [i,j] = H[i,j+1] - H[i,j]
            elif j == b-1 :
                H22 [i,j] = H[i,j-1] - H[i,j]

    return H11, H12, H21, H22

def hessian_adjoint (H) :
    a, b = np.shape (H)[1:3]

    H11 = np.zeros ((a,b))
    H12 = np.zeros ((a,b))
    H21 = np.zeros ((a,b))
    H22 = np.zeros ((a,b))

    for i in range (a) :
        for j in range (b) :
            if 0 < i < a-1 :
                H11 [i,j] = H[0,i-1,j] - 2 * H[0,i,j] + H[0,i+1,j]
            elif i == 0 :
                H11 [i,j] = H[0,i+1,j] - H[0,i,j]
            elif i == b-1 :
                H11 [i,j] = H[0,i-1,j] - H[0,i,j]

            if 0 < j < b-1 :
                H22 [i,j] = H[3,i,j-1] - 2 * H[3,i,j] + H[3,i,j+1]
            elif j == 0 :
                H22 [i,j] = H[3,i,j+1] - H[3,i,j]
            elif j == b-1 :
                H22 [i,j] = H[3,i,j-1] - H[3,i,j]

            if 0 < j < b-1 :
                if 0 < i < a-1 :
                    H12 [i,j] = H[1,i,j-1] - H[1,i,j] - H[1,i+1,j-1] + H[1,i+1,j]
                elif i == 0 :
                    H12 [i,j] = H[1,i+1,j] - H[1,i+1,j-1]
                elif i == a-1 :
                    H12 [i,j] = H[1,i,j-1] - H[1,i,j]
            if 0 < i < a-1 :
                if j == 0 :
                    H12 [i,j] = H[1,i+1,j] - H[1,i,j]
                elif j == b-1 :
                    H12 [i,j] = H[1,i,j-1] - H[1,i+1,j-1]
            if i == 0 :
                if j == 0 :
                    H12 [i,j] = H[1,i+1,j]
                elif j == b-1 :
                    H12 [i,j] = - H[1,i+1,j-1]
            if i == a-1 :
                if j == 0 :
                    H12 [i,j] = - H[1,i,j]
                elif j == b-1 :
                    H12 [i,j] = - H[1,i,j-1]

            if 0 < j < b-1 :
                if 0 < i < a-1 :
                    H21 [i,j] = H[2,i-1,j] - H[2,i,j] - H[2,i-1,j+1] + H[2,i,j+1]
                elif i == 0 :
                    H21 [i,j] = H[2,i,j+1] - H[2,i,j]
                elif i == a-1 :
                    H21 [i,j] = H[2,i-1,j] - H[2,i-1,j+1]

            if 0 < i < a-1 :
                if j == 0 :
                    H21 [i,j] = H[2,i,j+1] - H[2,i-1,j+1]
                elif j == b-1 :
                    H21 [i,j] = H[2,i-1,j] - H[2,i,j]

            if i == 0 :
                if j == 0 :
                    H21 [i,j] = H[2,i,j+1]
                elif j == b-1 :
                    H21 [i,j] = - H[2,i,j]

            if i == a-1 :
                if j == 0 :
                    H21 [i,j] = - H[2,i-1,j+1]
                elif j == b-1 :
                    H21 [i,j] = H[2,i-1,j]

    return H11 + H12 + H22 + H21
