include(LibFindMacros)

find_path(PETSc_INCLUDE_DIR
  NAMES petsc.h
)

find_library(PETSc_LIBRARY
  NAMES petsc
)

libfind_process(PETSc)
