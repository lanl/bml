include_directories(${CMAKE_SOURCE_DIR}/src/C-interface)

set(SOURCES_TYPED
  add_matrix_typed.c
  allocate_matrix_typed.c
  convert_matrix_typed.c
  copy_matrix_typed.c
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
  add_matrix.c
  allocate_matrix.c
  bml_test.c
  convert_matrix.c
  copy_matrix.c
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

add_test(allocate-dense-single_real bml-test -n allocate -t dense -p single_real)
add_test(allocate-dense-double_real bml-test -n allocate -t dense -p double_real)
add_test(allocate-dense-single_complex bml-test -n allocate -t dense -p single_complex)
add_test(allocate-dense-double_complex bml-test -n allocate -t dense -p double_complex)

add_test(add-dense-single_real bml-test -n add -t dense -p single_real)
add_test(add-dense-double_real bml-test -n add -t dense -p double_real)
add_test(add-dense-single_complex bml-test -n add -t dense -p single_complex)
add_test(add-dense-double_complex bml-test -n add -t dense -p double_complex)

add_test(multiply-dense-single_real bml-test -n multiply -t dense -p single_real)
add_test(multiply-dense-double_real bml-test -n multiply -t dense -p double_real)
add_test(multiply-dense-single_complex bml-test -n multiply -t dense -p single_complex)
add_test(multiply-dense-double_complex bml-test -n multiply -t dense -p double_complex)

add_test(convert-dense-single_real bml-test -n convert -t dense -p single_real)
add_test(convert-dense-double_real bml-test -n convert -t dense -p double_real)
add_test(convert-dense-single_complex bml-test -n convert -t dense -p single_complex)
add_test(convert-dense-double_complex bml-test -n convert -t dense -p double_complex)

add_test(copy-dense-single_real bml-test -n copy -t dense -p single_real)
add_test(copy-dense-double_real bml-test -n copy -t dense -p double_real)
add_test(copy-dense-single_complex bml-test -n copy -t dense -p single_complex)
add_test(copy-dense-double_complex bml-test -n copy -t dense -p double_complex)
