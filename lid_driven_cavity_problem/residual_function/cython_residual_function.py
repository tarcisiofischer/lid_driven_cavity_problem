try:
    from cython_.residual_function import residual_function
except ImportError:
    raise RuntimeError("Cython module is not compiled. Compile it by running command:\n \
    python setup.py build_ext --inplace\n \
    In the project's root directory.")

# Forward to Cython implementation
residual_function = residual_function
