all:
	g++ \
		-fPIC \
		-fopenmp \
		-O2 \
		-shared \
		-std=c++11 \
		-I ${CONDA_PREFIX}/include/python3.5m/ \
		`python-config --cflags --ldflags` \
        residual_function.cpp \
        residual_function_omp.cpp \
		_residual_function.cpp \
		-o _residual_function.so

