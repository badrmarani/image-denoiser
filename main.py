#!/usr/bin/env python3

# internal references
from source.optimization.chambolle import *
from source.utils.misc import bcolors

# third party
import cv2
from matplotlib.pyplot import *

# standard libraries
import os
import time
import argparse



pparam = dict(xlabel='Iterations', ylabel='SNR')


def run (max_iter = 100, mean = 0, sigma = 50,
            L = 60, step_size = 0.015625, epsilon = 1e-4) :

    os.chdir (dir := os.path.abspath('./data/imgs/'))

    for image in os.listdir(dir) :
        if image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg') :
            SNRall = []
            PSNRall = []

            name = os.path.splitext (image)[0]
            print(f'{bcolors.HEADER}Processing: {name}{bcolors.ENDC}')

            im = Image (image, max_iter = max_iter, mean = mean, sigma = sigma,
                        L = L, step_size = step_size, epsilon = epsilon)

            os.chdir (os.path.abspath('../results/'))
            cv2.imwrite (str(name) + '_' + 'gaussian-noise'+ '.jpg', im.noisy_image)

            for method in (methods := ['cham1', 'proj1', 'cham2', 'proj2']) :
                print(f'{bcolors.OKBLUE}>>>> Variational Method:{bcolors.ENDC}', method)

                DI, SNR, PSNR = im.denoise (ABS_IMG = image, method = method)

                SNRall.append (SNR)
                PSNRall.append (PSNR)

                # Save the results
                cv2.imwrite (str(name) + '_' + str(int(sigma)) + '-' + str(int(L)) + '_' + str(method)+ '.jpg', DI)
                os.chdir (os.path.abspath('../imgs/'))


            os.chdir (os.path.abspath('../results/'))
            with style.context(['science']):
                grid (True)
                for method, elt in zip(['cham1, proj1'], SNRall[0:2]):
                    plot (elt, label = str(method))

                legend(title = '$\max \|p^{n+1}-p^{n}\|_1$')
                autoscale(tight = True)
                savefig(str(name) + '_' + str(int(sigma)) + '-' + str(int(L)) + '_1' + '.jpg', dpi=300)
                close ()

                grid (True)
                for method, elt in zip(['cham2, proj2'], SNRall[2:4]):
                    plot (elt, label = str(method))

                legend(title = '$\max \|p^{n+1}-p^{n}\|_1$')
                autoscale(tight = True)
                savefig(str(name) + '_' + str(int(sigma)) + '-' + str(int(L)) + '_2' + '.jpg', dpi=300)
                close ()

            # print (f'{bcolors.HEADER}Results - ROF:{bcolors.ENDC}')
            # for elt in SNRall[0:2] :
            #     print (f'{bcolors.BOLD}SNR = {bcolors.ENDC}', elt[-1])
            #
            # print (f'{bcolors.HEADER}Results - ROF2:{bcolors.ENDC}')
            # for elt in SNRall[2:4] :
            #     print (f'{bcolors.BOLD}SNR = {bcolors.ENDC}', elt[-1])

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
