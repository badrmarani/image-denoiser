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
from src.utils.noise import gaussian
from src.utils.operators import div
from src.utils.operators import grad
from src.utils.operators import hessian
from src.utils.operators import hessian_adjoint

class Image () :
    def __init__ (self, im, mean = 0, sigma = 12, epsilon = 1e-2, L = 20,
                    step_size = 0.20, max_iter = 200) :

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

    def norm_in_Rn (self, u) :
        return np.sqrt (sum (np.power (u,2)))

    def norm_in_X (self, u) :
        return np.sum (np.power (u,2)) / np.product (np.shape (u))

    def norm_in_Y (self, u) :
        """ Reminder: $u \in Y = X \times X$
        """
        u1 = np.power (u [0], 2)
        u2 = np.power (u [1], 2)

        return np.sum (u1 + u2)

    def PSNR (self, u) :
        """ Peak Signal to Noise Ratio (PSNR)
        """
        x = np.power (np.divide (255,self.norm_in_X (self.image - u)), 2)
        return 10 * np.log10 (x)

    def SNR (self, u) :
        """ Signal to Noise Ratio (SNR)
        """
        return 10 * np.log10 (np.divide (self.norm_in_X (self.image), self.norm_in_X (self.image - u)))

    @property
    def chambolle1  (self) :
        g = self.noisy_image
        old_p = np.zeros (np.shape ((g,g)))

        snr = []

        for i in tqdm (range (self.max_iter), ascii = True) :
            gd = grad (div(old_p) - np.divide (g, self.L))


            # TODO: Test on Y norm
            norm_gd = self.norm_in_Rn (gd)

            new_p = np.divide (old_p + np.dot (self.tau,gd) , 1 + np.dot (self.tau, norm_gd))

            if (x := self.norm_in_Y (new_p - old_p)) < self.epsilon :
                print (f'\nNumber of iterations {i}'); break

            old_p = new_p


            ret = g - self.L * div (new_p)
            # snr.append (self.SNR (ret))

            snr.append (x)


            # os.chdir (os.path.abspath('../denoised/'))
            # if i in np.linspace (0, self.max_iter, 9, dtype = int) :
            #     cv2.imwrite ('iter_' + str(i) + '.jpg', ret)
            # os.chdir (os.path.abspath('../imgs/'))


        print (
            'PSNR is:', self.PSNR (ret),
            '\nSNR is:', self.SNR (ret)
        )


        return snr, ret

    @property
    def chambolle2  (self) :
        g = self.noisy_image

        snr = []

        old_p = np.zeros (np.shape ((g,g,g,g)))

        for i in tqdm (range (self.max_iter), ascii = True) :
            gd = hessian (hessian_adjoint(old_p) - np.divide (g, self.L))


            # TODO: Test on Y norm
            norm_gd = self.norm_in_Rn (gd)

            new_p = np.divide (old_p - np.dot (self.tau,gd) , 1 + self.tau*norm_gd)

            if (x := np.max (np.abs (new_p - old_p))) < self.epsilon :
                print (f'\nNumber of iterations {i}'); break

            old_p = new_p

            snr.append (x)


            ret = g - self.L * hessian_adjoint (new_p)

            # os.chdir (os.path.abspath('../denoised/'))
            # if i == 0 :
            #     cv2.imwrite ('face_iter_' + str(i) + '.jpg', ret)
            # os.chdir (os.path.abspath('../imgs/'))


        return snr, ret
