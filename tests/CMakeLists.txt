add_subdirectory(C-tests)
if(NOT BML_MPI)
  add_subdirectory(Fortran-tests)
endif()
