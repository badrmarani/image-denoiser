import matplotlib.pyplot as plt
import numpy as np

from source.optimization.chambolle import *
from source.utils.display import *

image = imread ('./data/imgs/girlface.jpg')


os.chdir (dir := os.path.abspath('./data/imgs/'))
im = Image ('girlface.jpg', max_iter = 10, mean = 0, sigma = 0,
            L = 20, step_size = 1/2, epsilon = 1e-2)


Y = im.runner ()


x = Plotter (Y , label  = ['sigma', 'PSNR'])
x.display (False)
