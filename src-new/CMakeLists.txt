add_subdirectory(Fortran-interface)
add_subdirectory(C-interface)

add_library(bml-new
  $<TARGET_OBJECTS:bml-dense>
  $<TARGET_OBJECTS:bml-dense-single_real>
  $<TARGET_OBJECTS:bml-dense-double_real>
  $<TARGET_OBJECTS:bml-ellpack>
  $<TARGET_OBJECTS:bml-ellpack-single_real>
  $<TARGET_OBJECTS:bml-ellpack-double_real>
  $<TARGET_OBJECTS:bml-fortran>
  $<TARGET_OBJECTS:bml-c>)

install(TARGETS bml-new
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
