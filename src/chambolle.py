import './utils/noise' as noise
import './utils/operators' as op

import numpy as np
from matplotlib.pyplot import *


img = 'lenna.jpg'
IMAGE_DIR = "../data/imgs/" + img


image = imread (IMAGE_DIR)
