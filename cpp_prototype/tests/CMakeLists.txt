include_directories(${CMAKE_SOURCE_DIR}/src)
add_executable(bml-test
    bml-test.cc)

add_test(NAME allocate COMMAND bml-test)
