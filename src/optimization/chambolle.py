#!/usr/bin/env python3

# standard libraries
import time
import os

# third party
import numpy as np
from matplotlib.pyplot import *
from tqdm import tqdm

# internal references
from src.utils.noise import gaussian
from src.utils.operators import div
from src.utils.operators import grad
from src.utils.operators import hessian
from src.utils.operators import hessian_adjoint

class Image () :
    def __init__ (self, im, mean = 0, sigma = 12, epsilon = 1e-2, L = 20,
                    step_size = 1/4, max_iter = 200) :

        if os.path.exists (dir := os.path.abspath(im)) :
            self.image = imread (dir)
        else :
            print (dir)
            raise ValueError('Image is not found.')

        self.mean = mean
        self.sigma = sigma
        self.epsilon = epsilon
        self.L = L
        self.tau = step_size
        self.max_iter = max_iter

        self.noisy_image = gaussian (self.image, self.mean, self.sigma)

    def norm (self, u) :
        return np.sqrt (sum (np.power (u,2)))

    def SNR (self, u) :
        """ Signal to Noise Ratio (SNR)
        """
        return 20 * np.log10 (np.divide (norm(u), norm(self.noisy_image - u)))

    def chambolle1  (self) :
        g = self.noisy_image
        old_p = np.zeros (np.shape ((g,g)))

        start = time.time ()
        for i in tqdm (range (self.max_iter), ascii = True) :
            gd = grad (div(old_p) - np.divide (g, self.L))

            norm_gd = self.norm(gd)

            new_p = np.divide (old_p + np.dot (self.tau,gd) , 1 + self.tau*norm_gd)

            if (x := np.max (np.abs (new_p - old_p))) < self.epsilon :
                print (f'\nNumber of iterations {i}'); break

            old_p = new_p

        return g - self.L * div (new_p)

    def chambolle2  (self) :
        g = self.noisy_image
        old_p = np.zeros (np.shape ((g,g,g,g)))

        start = time.time ()
        for i in tqdm (range (self.max_iter), ascii = True) :
            gd = hessian (hessian_adjoint(old_p) - np.divide (g, self.L))

            norm_gd = self.norm(gd)

            new_p = np.divide (old_p + np.dot (self.tau,gd) , 1 + self.tau*norm_gd)

            if (np.max (np.abs (new_p - old_p))) < self.epsilon :
                print (f'\nNumber of iterations {i}'); break

            old_p = new_p

        return g - self.L * hessian_adjoint (new_p)
