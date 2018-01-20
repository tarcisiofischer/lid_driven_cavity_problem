try:
    import _residual_function
except ImportError:
    raise RuntimeError("C++ module is not compiled.")


# Forward to C++ implementation
residual_function = _residual_function.residual_function_omp
