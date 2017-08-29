try:
    import _residual_function
except ImportError:
    raise RuntimeError("C++ module is not compiled. Compile it by running command:\n \
    make\n \
    In the c++/ directory.")


# Forward to C++ implementation
residual_function = _residual_function.residual_function_omp
