include_directories(${PROJECT_SOURCE_DIR}/src)
add_executable(bml-test
    bml-test.cc)
target_link_libraries(bml-test bml)
add_test(NAME allocate
    COMMAND bml-test)
