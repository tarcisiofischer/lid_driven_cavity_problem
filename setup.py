from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
import numpy


def get_cython_modules():
    return [
        Extension(
            'cython_.residual_function',
            sources=['cython_/residual_function.pyx'],
            language='c++',
            include_dirs=[numpy.get_include()],
        ),
    ]

setup(
    ext_modules=cythonize(get_cython_modules()),
    package_dir={'': ''},
)
