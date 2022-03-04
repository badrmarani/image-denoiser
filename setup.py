#!/usr/bin/env python3

# third party
from setuptools.command.install import install
from setuptools import setup, find_packages

# standard libraries
import io
import os


here = os.path.abspath(os.path.dirname(__file__))
readmefile = os.path.join(here, 'README.md')

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    mode = kwargs.get('mode', 'r')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        if 'b' not in mode:
            with io.open(filename, mode=mode, encoding=encoding) as f:
                buf.append(f.read())
        else:
            with io.open(filename, mode=mode) as f:
                buf.append(f.read())
    return sep.join(buf)

class image_install (install):
    def run(self):
        mode = None
        while mode not in ['', 'install']:
            mode = input("Installation mode: [develop]/install/cancel: ")
        if mode == 'install':
            return install.run(self)

setup (
    name = 'Image',
    description = read (readmefile),
    long_description = read (readmefile),
    url = 'https://github.com/maranibadr/image',
    classifiers = [
        'Programming Language :: Python',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries :: Application Frameworks'
    ],

    install_requires=["numpy", "matplotlib", "opencv-python", "tqdm"],

    packages=find_packages(),
    package_data={},
)
