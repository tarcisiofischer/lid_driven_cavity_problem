import numpy
from Cython.Build import cythonize

from distutils.core import setup
from distutils.extension import Extension

def get_cython_modules():
    return [
        Extension(
            'lid_driven_cavity_problem.residual_function.cython_residual_function',
            sources=['lid_driven_cavity_problem/residual_function/cython_residual_function.pyx'],
            language='c++',
            include_dirs=[numpy.get_include()],
        ),
    ]

setup(
    ext_modules=cythonize(get_cython_modules()),
    package_dir={'': ''},
)
