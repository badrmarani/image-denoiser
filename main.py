#!/usr/bin/env python3


# third party
import cv2
from matplotlib.pyplot import *
from tabulate import tabulate

# standard libraries
import os
import time
import argparse

# internal references
from source.optimization.chambolle import *
from source.utils.display import Plotter

from source.utils.misc import bcolors

def run (max_iter, mean, sigma ,
            L, step_size, epsilon) :

    os.chdir (dir := os.path.abspath('./data/imgs/'))

    for image in os.listdir(dir) :
        if image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg') :
            name = os.path.splitext (image)[0]

            print(f'{bcolors.HEADER}Processing: {name}{bcolors.ENDC}')
            im = Image (image, max_iter = max_iter, mean = mean, sigma = sigma,
                        L = L, step_size = step_size, epsilon = epsilon)

            os.chdir (os.path.abspath('../results/'))
            cv2.imwrite (str(name) + '-GN-S(' + str(int(sigma)) + ').jpg', im.noisy_image)

            for method in (methods := ['cham1']) :
                print(
                    f'{bcolors.OKBLUE}>>>> Variational Method:{bcolors.ENDC}', method,
                    '\n', tabulate(
                        [[max_iter, mean, sigma, L, step_size, epsilon]],
                        headers=['Number of iterations', 'Mean', 'Noise', 'Lambda', 'Step size', 'Epsilon']), '\n'
                )

                DI, CRIT, SNR, PSNR, sigma, L = im.denoise (ABS_IMG = image, method = method)

                # Save the results
                cv2.imwrite (str(name) + '-' + str(method).upper() + '-S(' + str(int(sigma)) + ')-L(' + str(int(L)) + ').jpg', DI)
                os.chdir (os.path.abspath('../imgs/'))

                #
                # Here, I test each method alone.
                #

                # print (PSNR [-1])

                # # SNR
                # PLOT = Plotter(
                #     PSNR,
                #     title = 'PSNR',
                #     xaxis = 'Iterations', yaxis = 'SNR',
                #     label = [method]
                # ).display (save = True, name = 'test')

                # # $ \max (|p^{n+1} - p^{n}|) < \epsilon $
                # PLOT = Plotter(
                #     SNR,
                #     title = 'SNR',
                #     xaxis = 'Iterations', yaxis = 'SNR',
                #     label = [method]
                # ).display (save = True, name = 'test')

            os.chdir (os.path.abspath('../imgs/'))


def main() :
    parser = argparse.ArgumentParser(description = 'Variational Image Denoising')
    parser.add_argument('--max-iter', default = 200, type = int)
    parser.add_argument('--mean', default = 0, type = float)
    parser.add_argument('--sigma', default = 50, type = float)
    parser.add_argument('--L', default = 60, type = float)
    parser.add_argument('--step-size', default = .20, type = float)
    parser.add_argument('--epsilon', default = 1e-4, type = float)
    args = parser.parse_args()

    run(**vars(args))


if __name__ == '__main__' :
    main()
