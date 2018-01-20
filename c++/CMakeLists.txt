find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

add_library(
    residual_function
    SHARED
        residual_function.cpp
        residual_function_omp.cpp
        _residual_function.cpp
)

target_link_libraries(
	residual_function
	PRIVATE
		pybind11::pybind11
)

set(
    CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}"
)
set_target_properties(
	residual_function
	PROPERTIES
		OUTPUT_NAME _residual_function
		PREFIX ""
)

install(
	TARGETS residual_function
	DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/
)
