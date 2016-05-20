__author__ = 'chaitanya'

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("/users/chaitanya/PyCharmProjects/EventRec/bj.pyx")
)