# internal references
from src.optimization.chambolle import *

# third party
import cv2
from matplotlib.pyplot import *

# standard libraries
import os
import time

def main () :

    i = 0

    os.chdir (dir := os.path.abspath('./data/imgs/'))

    for image in os.listdir(dir) :
        if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg") :

            print ('Processing :', image)

            im = imread (image)
            if len (np.shape (im)) != 2 :
                b, g, r = cv2.split (im)

                cv2.imwrite (x := 'image' + str(i) + '_x_' + '.jpg', b)
                cv2.imwrite (y := 'image' + str(i) + '_y_' + '.jpg', g)
                cv2.imwrite (z := 'image' + str(i) + '_z_' + '.jpg', r)

                start = time.time ()

                print ('First layer ...');
                DIx = Image (x).chambolle1(); os.remove (x)

                print ('Second layer ...');
                DIy = Image (y).chambolle1(); os.remove (y)

                print ('third layer ...');
                DIz = Image (z).chambolle1(); os.remove (z)

                DI = cv2.merge ([DIz,DIy,DIx])

            else :
                DI = Image (image).chambolle1()

            print (f'\nTime spent is: {time.time() - start}')
            os.chdir (os.path.abspath('../denoised/'))

            cv2.imwrite ('image' + str(i) + '.jpg', DI); i += 1
            os.chdir (os.path.abspath('../imgs/'))

if __name__ == '__main__':
    main()
