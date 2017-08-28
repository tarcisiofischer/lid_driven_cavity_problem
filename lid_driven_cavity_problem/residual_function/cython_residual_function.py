try:
    # Forward to Cython implementation
    from cython_.residual_function import residual_function  # @UnusedImport
except ImportError:
    raise RuntimeError("Cython module is not compiled. Compile it by running command:\n \
    python setup.py build_ext --inplace\n \
    In the project's root directory.")
