find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PETSc REQUIRED)

add_library(
    residual_function
    SHARED
        _residual_function.cpp

        residual_function.cpp
        residual_function_omp.cpp
)

add_library(
	petsc_helpers
	SHARED
		_petsc_helpers.cpp

        petsc_solver_helpers.hpp
        petsc_solver_helpers.cpp
)		

include_directories(
    ${PETSc_INCLUDE_DIRS}
)

target_link_libraries(
	residual_function
	PRIVATE
		pybind11::pybind11
)

target_link_libraries(
	petsc_helpers
	PRIVATE
		pybind11::pybind11
		${PETSc_LIBRARY}
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
set_target_properties(
	petsc_helpers
	PROPERTIES
		OUTPUT_NAME _petsc_helpers
		PREFIX ""
)

install(
	TARGETS residual_function petsc_helpers
	DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/
)
