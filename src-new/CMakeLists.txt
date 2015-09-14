add_subdirectory(Fortran-interface)
add_subdirectory(C-interface)

add_library(bml-new
  $<TARGET_OBJECTS:bml-dense>
  $<TARGET_OBJECTS:bml-fortran>
  $<TARGET_OBJECTS:bml-c>)

install(TARGETS bml-new
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
