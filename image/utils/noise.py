#!/usr/bin/env python3

# third party
import numpy as np

def gaussian (im, mean = 0, sigma = 20) :
    return im + np.random.normal (mean, sigma, np.shape (im))
