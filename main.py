# internal references
from src.optimization.chambolle import *

# third party
import cv2
from matplotlib.pyplot import *

# standard libraries
import os
import time
import argparse

def run (max_iter = 100, mean = 0, sigma = 50, L = 60, step_size = 0.015625, epsilon = 1e-4) :

    i = 0

    os.chdir (dir := os.path.abspath('./data/imgs/'))
    for image in os.listdir(dir) :
        if image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg') :

            print ('Processing :', image)
            start = time.time ()

            im = imread (image)

            if len (np.shape (im)) != 2 :
                b, g, r = cv2.split (im)

                cv2.imwrite (x := 'image' + str(i) + '_x_' + '.jpg', b)
                cv2.imwrite (y := 'image' + str(i) + '_y_' + '.jpg', g)
                cv2.imwrite (z := 'image' + str(i) + '_z_' + '.jpg', r)

                print ('First layer ...');
                DIx = Image (x, max_iter = max_iter, mean = mean,
                                sigma = sigma, L = L, step_size = step_size,
                                epsilon = epsilon).chambolle2 [-1]; os.remove (x)

                print ('Second layer ...');
                DIy = Image (y, max_iter = max_iter, mean = mean,
                                sigma = sigma, L = L, step_size = step_size,
                                epsilon = epsilon).chambolle2 [-1]; os.remove (y)
                print ('Third layer ...');
                DIz = Image (z, max_iter = max_iter, mean = mean,
                                sigma = sigma, L = L, step_size = step_size,
                                epsilon = epsilon).chambolle2 [-1]; os.remove (z)

                DI = cv2.merge ([DIz,DIy,DIx])

            else :
                cham1 = Image (image, max_iter = max_iter, mean = mean, sigma = sigma, L = L, step_size = step_size, epsilon = epsilon).chambolle2
                DI = cham1 [-1]

            # print (f'\nTime spent is: {time.time() - start}')
            os.chdir (os.path.abspath('../denoised/'))

            cv2.imwrite ('image2-' + str(sigma) + '-' + str(L) + '.jpg', DI); i += 1
            os.chdir (os.path.abspath('../imgs/'))

            # SNR = cham1 [0]
            # with style.context(['science']):
            #     plot (SNR); grid ()
            #     show ()
                #     # savefig('fig1.pdf')
                #     # savefig('fig1.jpg', dpi=300)
            break

def main():
    parser = argparse.ArgumentParser(description = 'Variational Image Denoising')
    parser.add_argument('--max-iter', default = 200, type = int)
    parser.add_argument('--mean', default = 0, type = float)
    parser.add_argument('--sigma', default = 50, type = float)
    parser.add_argument('--L', default = 60, type = float)
    parser.add_argument('--step-size', default = .20, type = float)
    parser.add_argument('--epsilon', default = 1e-4, type = float)
    args = parser.parse_args()
    run(**vars(args))

if __name__ == '__main__':
    main()
