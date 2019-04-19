add_subdirectory(Fortran-interface)
add_subdirectory(C-interface)
add_subdirectory(internal-blas)

add_library(bml
  $<TARGET_OBJECTS:bml-c>
  $<TARGET_OBJECTS:bml-internal-blas-double_complex>
  $<TARGET_OBJECTS:bml-internal-blas-double_real>
  $<TARGET_OBJECTS:bml-internal-blas-single_complex>
  $<TARGET_OBJECTS:bml-internal-blas-single_real>
  $<TARGET_OBJECTS:bml-dense-double_complex>
  $<TARGET_OBJECTS:bml-dense-double_real>
  $<TARGET_OBJECTS:bml-dense-single_complex>
  $<TARGET_OBJECTS:bml-dense-single_real>
  $<TARGET_OBJECTS:bml-dense>
  $<TARGET_OBJECTS:bml-ellpack-double_complex>
  $<TARGET_OBJECTS:bml-ellpack-double_real>
  $<TARGET_OBJECTS:bml-ellpack-single_complex>
  $<TARGET_OBJECTS:bml-ellpack-single_real>
  $<TARGET_OBJECTS:bml-ellpack>
  $<TARGET_OBJECTS:bml-ellblock-double_complex>
  $<TARGET_OBJECTS:bml-ellblock-double_real>
  $<TARGET_OBJECTS:bml-ellblock-single_complex>
  $<TARGET_OBJECTS:bml-ellblock-single_real>
  $<TARGET_OBJECTS:bml-ellblock>
  $<TARGET_OBJECTS:bml-ellsort-double_complex>
  $<TARGET_OBJECTS:bml-ellsort-double_real>
  $<TARGET_OBJECTS:bml-ellsort-single_complex>
  $<TARGET_OBJECTS:bml-ellsort-single_real>
  $<TARGET_OBJECTS:bml-ellsort>)
set_target_properties(bml
  PROPERTIES
  SOVERSION ${PROJECT_VERSION}
  POSITION_INDEPENDENT_CODE yes)

add_library(bml_fortran
  $<TARGET_OBJECTS:bml-fortran>)
set_target_properties(bml_fortran
  PROPERTIES
  SOVERSION ${PROJECT_VERSION}
  POSITION_INDEPENDENT_CODE yes)

install(TARGETS bml bml_fortran
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
