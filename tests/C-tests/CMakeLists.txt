include_directories(${PROJECT_SOURCE_DIR}/src/C-interface)

if(BML_OMP_OFFLOAD)
  add_executable(test-openmp_offload test_openmp_offload.c)
  target_link_libraries(test-openmp_offload bml ${LINK_LIBRARIES})
  set_target_properties(test-openmp_offload
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS}
    LINK_FLAGS ${OpenMP_C_FLAGS})
  add_test(openmp_offload ${BML_NONMPI_PRECOMMAND} ${BML_NONMPI_PRECOMMAND_ARGS}
           ${CMAKE_CURRENT_BINARY_DIR}/test-openmp_offload)
endif()

if(BML_MAGMA)
  add_executable(test-magma test_magma.c)
  target_link_libraries(test-magma ${LINK_LIBRARIES})
  set_target_properties(test-magma
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS}
    LINK_FLAGS ${OpenMP_C_FLAGS})
  add_test(test-magma ${BML_NONMPI_PRECOMMAND} ${BML_NONMPI_PRECOMMAND_ARGS}
           ${CMAKE_CURRENT_BINARY_DIR}/test-magma)
endif()

set(SOURCES_TYPED
  add_matrix_typed.c
  adjacency_matrix_typed.c
  adjungate_triangle_matrix_typed.c
  allocate_matrix_typed.c
  convert_matrix_typed.c
  copy_matrix_typed.c
  diagonalize_matrix_typed.c
  get_element_typed.c
  get_set_diagonal_typed.c
  get_sparsity_typed.c
  import_export_matrix_typed.c
  introspection_typed.c
  inverse_matrix_typed.c
  io_matrix_typed.c
  mpi_sendrecv_typed.c
  multiply_banded_matrix_typed.c
  multiply_matrix_typed.c
  element_multiply_matrix_typed.c
  multiply_matrix_x2_typed.c
  normalize_matrix_typed.c
  norm_matrix_typed.c
  print_matrix_typed.c
  scale_matrix_typed.c
  set_element_typed.c
  set_row_typed.c
  submatrix_matrix_typed.c
  bml_gemm_typed.c
  trace_mult_typed.c
  threshold_matrix_typed.c
  trace_matrix_typed.c
  transpose_matrix_typed.c)

include(${PROJECT_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)
bml_add_typed_library(bmltests single_real "${SOURCES_TYPED}")
bml_add_typed_library(bmltests double_real "${SOURCES_TYPED}")
if(BML_COMPLEX)
  bml_add_typed_library(bmltests single_complex "${SOURCES_TYPED}")
  bml_add_typed_library(bmltests double_complex "${SOURCES_TYPED}")
endif()

if(BML_COMPLEX)
  add_library(bmltests
    $<TARGET_OBJECTS:bmltests-single_real>
    $<TARGET_OBJECTS:bmltests-double_real>
    $<TARGET_OBJECTS:bmltests-single_complex>
    $<TARGET_OBJECTS:bmltests-double_complex>)
else()
  add_library(bmltests
    $<TARGET_OBJECTS:bmltests-single_real>
    $<TARGET_OBJECTS:bmltests-double_real>)
endif()
set_target_properties(bmltests
  PROPERTIES
  POSITION_INDEPENDENT_CODE yes)

add_executable(bml-test
  add_matrix.c
  adjacency_matrix.c
  adjungate_triangle_matrix.c
  allocate_matrix.c
  bml_test.c
  convert_matrix.c
  copy_matrix.c
  diagonalize_matrix.c
  get_element.c
  get_set_diagonal.c
  get_sparsity.c
  import_export_matrix.c
  introspection.c
  inverse_matrix.c
  io_matrix.c
  mpi_sendrecv.c
  multiply_banded_matrix.c
  multiply_matrix.c
  element_multiply_matrix.c
  multiply_matrix_x2.c
  normalize_matrix.c
  norm_matrix.c
  print_matrix.c
  scale_matrix.c
  set_element.c
  set_row.c
  submatrix_matrix.c
  bml_gemm.c
  trace_mult.c
  threshold_matrix.c
  trace_matrix.c
  transpose_matrix.c)

message(STATUS "tests: LINK_LIBRARIES=${LINK_LIBRARIES}")
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

message(STATUS "BML_NONMPI_PRECOMMAND=${BML_NONMPI_PRECOMMAND}")
message(STATUS "BML_NONMPI_PRECOMMAND_ARGS=${BML_NONMPI_PRECOMMAND_ARGS}")
if(BML_COMPLEX)
  set(precisions single_real double_real single_complex double_complex)
else()
  set(precisions single_real double_real)
endif()
function(test_formats ${formats})
  foreach(T ${formats} )
    foreach(P ${precisions})
      add_test(NAME C-${N}-${T}-${P}
        COMMAND ${BML_NONMPI_PRECOMMAND} ${BML_NONMPI_PRECOMMAND_ARGS}
        ${CMAKE_CURRENT_BINARY_DIR}/bml-test -n ${N} -t ${T} -p ${P})
      if(NOT BML_MPI AND NOT BML_OPENMP AND VALGRIND AND BML_VALGRIND)
        add_test(C-${N}-${T}-${P}-valgrind ${VALGRIND} ${VALGRIND_COMMON_ARGS}
          ${CMAKE_CURRENT_BINARY_DIR}/bml-test -n ${N} -t ${T} -p ${P})
      endif()
    endforeach()
  endforeach()
endfunction(test_formats)

function(test_formats_mpi ${formats})
  foreach(T ${formats} )
    foreach(P ${precisions})
      add_test(MPI-C-${N}-${T}-${P}
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROCS_FLAG} ${MPIEXEC_NUMPROCS} ${MPIEXEC_PREFLAGS}
        ${CMAKE_CURRENT_BINARY_DIR}/bml-test -n ${N} -t ${T} -p ${P})
    endforeach()
  endforeach()
endfunction(test_formats_mpi)

set(testlist-mpi
    add
    allocate
    convert
    copy
    diagonalize
    get_sparsity
    import_export
    introspection
    io_matrix
    scale
    mpi_sendrecv
    multiply
    norm
    normalize
    print
    set_row
    threshold
    trace
    transpose
)

set(testlist
  add
  adjacency
  adjungate_triangle
  allocate
  bml_gemm
  convert
  copy
  diagonalize
  get_element
  get_set_diagonal
  get_sparsity
  import_export
  introspection
  inverse
  io_matrix
  multiply
  element_multiply
  multiply_banded
  multiply_x2
  norm
  normalize
  print
  scale
  set_element
  set_row
  submatrix
  threshold
  trace
  trace_mult
  transpose
)

foreach(N ${testlist})
  set(formats dense ellpack ellblock csr)
  # Setting specific formats for a particular bml function
  if(${N} MATCHES diagonalize)
    set(formats dense ellpack ellblock csr)
  endif()
  if(${N} MATCHES inverse)
    set(formats dense ellpack ellblock csr)
  endif()
  if(${N} MATCHES set_element)
    set(formats dense ellpack ellblock csr)
  endif()
  if(${N} MATCHES element_multiply)
    set(formats dense ellpack csr)
  endif()

  test_formats(${formats})
endforeach()

if(BML_MPI)
  if(BML_SCALAPACK)
    foreach(N ${testlist-mpi})
      set(formats dense ellpack ellblock csr)
      test_formats_mpi(${formats})
    endforeach()
  else()
    foreach(N ${testlist-mpi})
      if(NOT ${N} MATCHES diagonalize)
        set(formats dense ellpack ellblock csr)
        test_formats_mpi(${formats})
      endif()
    endforeach()
  endif()
endif()

add_executable(test-backtrace test_backtrace.c)
target_link_libraries(test-backtrace bml ${LINK_LIBRARIES})
set_target_properties(test-backtrace
  PROPERTIES
  LINK_FLAGS "--coverage")
if(OPENMP_FOUND)
  set_target_properties(test-backtrace
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS}
    LINK_FLAGS ${OpenMP_C_FLAGS})
endif()
add_test(backtrace test-backtrace)
if(ADDR2LINE)
  set_tests_properties(backtrace
    PROPERTIES
    # For some compiler combinations source line resolution is not working. In
    # this case the backtrace will print out something like '??:0' which makes
    # this test not very useful unfortunately.
    PASS_REGULAR_EXPRESSION "test_backtrace.c:6[\\n]*.*test_backtrace.c:12;[?]+:[0?]")
else()
  set_tests_properties(backtrace
    PROPERTIES
    PASS_REGULAR_EXPRESSION "test-backtrace[(]foo.*[\\n]*.*test-backtrace[(]main;[?]+:[0?]")
endif()
