include_directories(${CMAKE_SOURCE_DIR}/src/C-interface)

include(${CMAKE_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)

set(SOURCES_TYPED
  multiply_matrix.c)

bml_add_typed_library(bmltests single_real ${SOURCES_TYPED})
add_library(bmltests
  $<TARGET_OBJECTS:bmltests-single_real>)
if(OPENMP_FOUND)
  set_target_properties(bmltests-single_real
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()
if(MPI_C_FOUND AND MPI_C_COMPILE_FLAGS)
  set_target_properties(bmltests-single_real
    PROPERTIES COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
endif()

add_executable(bml-test
  bml_test.c)
target_link_libraries(bml-test bmltests bml ${LINK_LIBRARIES})
set_target_properties(bml-test
  PROPERTIES
  LINK_FLAGS "-fprofile-arcs -lgcov")
if(OPENMP_FOUND)
  set_target_properties(bml-test
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS}
    LINK_FLAGS ${OpenMP_C_FLAGS})
endif()
if(MPI_C_FOUND AND MPI_C_COMPILE_FLAGS)
  set_target_properties(bml-test
    PROPERTIES
    COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS}
    LINK_FLAGS ${MPI_C_LINK_FLAGS})
endif()

add_test(dummy /bin/true)
