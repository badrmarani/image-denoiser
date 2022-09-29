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
from image.utils.noise import gaussian
from image.utils.operators import div
from image.utils.operators import grad
from image.utils.operators import hessian
from image.utils.operators import hessian_adjoint

from image.optimization.projection import projection1 as proj1
from image.optimization.projection import projection2 as proj2

class Image () :
    def __init__ (self, im, mean = 0, sigma = 12, epsilon = 1e-2, L = 20,
                    step_size = 0.20, max_iter = 100) :

        self.name = os.path.splitext (im)[0]
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

        self.snr = []
        self.psnr = []

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

    def chambolle1  (self, K = None) :
        g = self.noisy_image

        old_p = np.zeros (np.shape ((g,g)))

        RET = g

        CRIT = []
        SNR = []
        PSNR = []

        for i in tqdm (range (self.max_iter), ascii = True) :

            gd = grad (div(old_p) - np.divide (g, self.L))
            norm_gd = self.norm_in_Rn (gd)

            new_p = np.divide (old_p + np.dot (self.tau,gd) , 1 + np.dot (self.tau, norm_gd))

            if (x := self.norm_in_Y (new_p - old_p)) < self.epsilon :
                break

            old_p = new_p

            RET = g - self.L * div (new_p)

            CRIT.append (x)
            SNR.append (self.SNR (RET))
            PSNR.append (self.PSNR (RET))

        return RET, CRIT, SNR, PSNR

    def runner (self) :

        r, c, s = [0], [0], [0]
        P = [0]

        PSNR = []

        Sigma = np.linspace (0, 100, 5, dtype = int)
        for s in Sigma :
            r, c, s, P = self.chambolle1 ()

            PSNR.append (P)

        return Sigma, PSNR

    def chambolle2  (self, K = None) :
        # if K is not None :
        #     # In case we're working on RBG images
        #     self.image = imread (K)
        #     self.noisy_image = gaussian (self.image, self.mean, self.sigma)

        g = self.noisy_image
        old_p = np.zeros (np.shape ((g,g,g,g)))

        RET = g

        CRIT = []
        SNR = []
        PSNR = []

        for i in tqdm (range (self.max_iter), ascii = True) :
            gd = hessian (hessian_adjoint(old_p) - np.divide (g, self.L))
            norm_gd = self.norm_in_Rn (gd)

            new_p = np.divide (old_p - np.dot (self.tau,gd) , 1 + self.tau*norm_gd)

            if (x := self.norm_in_Y (new_p - old_p)) < self.epsilon :
                break

            old_p = new_p

            RET = g - self.L * hessian_adjoint (new_p)

            CRIT.append (x)
            SNR.append (self.SNR (RET))
            PSNR.append (self.PSNR (RET))

        return RET, CRIT, SNR, PSNR

    def projection1 (self) :
        CRIT = [0]
        SNR = [0]
        PSNR = [0]

        RET, CRIT, SNR, PSNR = proj1 (image = self.noisy_image, mean = self.mean,
                            sigma = self.sigma, epsilon = self.epsilon,
                            L = self.L, step_size = self.tau, max_iter = self.max_iter)

        return RET, CRIT, PSNR

    def projection2 (self) :
        CRIT = [0]
        SNR = [0]
        PSNR = [0]

        RET, CRIT, SNR, PSNR = proj2 (image = self.noisy_image, mean = self.mean,
                            sigma = self.sigma, epsilon = self.epsilon,
                            L = self.L, step_size = self.tau, max_iter = self.max_iter)

        return RET, CRIT, PSNR

    def denoise (self, ABS_IMG, method) :
        denoisers = {
            'cham1' : self.chambolle1,
            'proj1' : self.projection1,
            'cham2' : self.chambolle2,
            'proj2' : self.projection2,
        }

        if method not in denoisers.keys() :
            raise ValueError('Unfortunately, this method is not yet implemented...')
        else :
            if method == 'cham1' :
                self.L = 60
            elif method == 'cham2' :
                self.L = 50
                self.epsilon = 1e-4
                self.tau = 0.015625
            elif method == 'proj1' :
                self.L = 15
                self.epsilon = 1
            elif method == 'proj2' :
                self.L = 15

            # os.chdir (dir := os.path.abspath('../imgs/'))
            self.image = imread (ABS_IMG)

            if len (np.shape (self.image)) != 2 :
                x, y, z = cv2.split (self.image)

                w = []
                DI = []
                for i, elt in zip(range(1,4), [x,y,z]) :
                    print ('> layer ::', i); cv2.imwrite (a := 'X.jpg', elt)

                    self.image = elt
                    self.noisy_image = gaussian (self.image, self.mean, self.sigma)

                    DI.append (denoisers [method] () [0]); os.remove (a)
                    w.append (self.noisy_image)

                w.reverse()

                # os.chdir (os.path.abspath('../results/'))
                cv2.imwrite (str (self.name) + '_gaussian-noise.jpg', cv2.merge (w))

                DI.reverse()
                DI = cv2.merge (DI)

                try :
                    SNR = [self.SNR (DI)]
                    PSNR = [self.SNR (DI)]
                except :
                    SNR = []
                    PSNR = []
            else :
                # os.chdir (dir := os.path.abspath('../results/'))
                DI, CRIT, SNR, PSNR = denoisers [method] ()

            CRIT = [0]
            return DI, CRIT, SNR, PSNR, self.sigma, self.L
