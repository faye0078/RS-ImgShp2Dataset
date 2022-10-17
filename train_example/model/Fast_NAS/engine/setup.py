from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("miou_utils.pyx"),
    include_dirs = [numpy.get_include()],
)