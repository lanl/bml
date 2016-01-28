include_directories(${CMAKE_SOURCE_DIR}/src/C-interface)

set(SOURCES_TYPED
  multiply_matrix_typed.c)

include(${CMAKE_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)
bml_add_typed_library(bmltests single_real "${SOURCES_TYPED}")
bml_add_typed_library(bmltests double_real "${SOURCES_TYPED}")
bml_add_typed_library(bmltests single_complex "${SOURCES_TYPED}")
bml_add_typed_library(bmltests double_complex "${SOURCES_TYPED}")

add_library(bmltests
  $<TARGET_OBJECTS:bmltests-single_real>
  $<TARGET_OBJECTS:bmltests-double_real>
  $<TARGET_OBJECTS:bmltests-single_complex>
  $<TARGET_OBJECTS:bmltests-double_complex>)

if(OPENMP_FOUND)
  set_target_properties(bmltests-single_real
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()
if(MPI_C_FOUND AND MPI_C_COMPILE_FLAGS)
  set_target_properties(bmltests-single_real
    PROPERTIES COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
endif()

add_executable(bml-test
  bml_test.c
  multiply_matrix.c)
target_link_libraries(bml-test bmltests bml ${LINK_LIBRARIES})
set_target_properties(bml-test
  PROPERTIES
  LINK_FLAGS "--coverage")
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

add_test(multiply-dense-single_real bml-test -n multiply -t dense -p single_real)
add_test(multiply-dense-double_real bml-test -n multiply -t dense -p double_real)
add_test(multiply-dense-single_complex bml-test -n multiply -t dense -p single_complex)
add_test(multiply-dense-double_complex bml-test -n multiply -t dense -p double_complex)
