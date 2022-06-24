#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

message(STATUS "Compiling BML Tensor core multiply...")


include_directories(${PROJECT_SOURCE_DIR}/src/C-interface/dense/mptc)

set(HEADERS-MPTC
  bml_mptc_dense.cuh)

set(SOURCES-MPTC
  bml_mptc_dense.cu)

add_library(bml-dense-mptc OBJECT ${SOURCES-MPTC})
set_target_properties(bml-dense-mptc
  PROPERTIES
  POSITION_INDEPENDENT_CODE yes)


# Compiling cuda files to make a C++ library
#cuda_add_library(mptc 
#  ${PROJECT_SOURCE_DIR}/src/C-interface/dense/mptc/bml_mptc_dense.cu)  
#  ${PROJECT_SOURCE_DIR}/src/sp2_tensorcore/src/tcore_hp_emulator.cu
#  ${PROJECT_SOURCE_DIR}/src/sp2_tensorcore/src/linalg_tools.cu)

# Precompile cuda objects for building the FORTRAN library
# This is not currently workin, but it is the way cuda object
# files should be created.

#cuda_compile(MPTC_O
#  ${PROJECT_SOURCE_DIR}/src/sp2_tensorcore/src/prg_sp2_tensorcore.cu  
#  ${PROJECT_SOURCE_DIR}/src/sp2_tensorcore/src/tcore_hp_emulator.cu
#  ${PROJECT_SOURCE_DIR}/src/C-interface/dense/mptc/bml_mptc_dense.cu)



# Gathering cuda objects. Instead we are manually uning the objects
# that were generated when prg_sp2_tc library was added.
#set(SP2_TC_O 
#"${PROJECT_BINARY_DIR}/src/sp2_tensorcore/CMakeFiles/prg_sp2_tc.dir/src/prg_sp2_tc_generated_prg_sp2_tensorcore.cu.o"
#"${PROJECT_BINARY_DIR}/src/sp2_tensorcore/CMakeFiles/prg_sp2_tc.dir/src/prg_sp2_tc_generated_linalg_tools.cu.o" 
#"${PROJECT_BINARY_DIR}/src/sp2_tensorcore/CMakeFiles/prg_sp2_tc.dir/src/prg_sp2_tc_generated_tcore_hp_emulator.cu.o" 
#)

# Making the Fortran lib out of the precompiled cuda objects.
#add_library(prg_sp2_tc_fortran
#  ${PROJECT_SOURCE_DIR}/src/sp2_tensorcore/src/prg_sp2_tensorcore_mod.F90
#  ${SP2_TC_O})

#set(TCSP2_INCLUDE "${PROJECT_SOURCE_DIR}/src/sp2_tensorcore/src") 

#message(STATUS "Including the following directories" ${TCSP2_INCLUDE})

#target_include_directories(prg_sp2_tc PUBLIC ${TCSP2_INCLUDE} ${CUDA_INCLUDE_DIRS}  ${CUDA_CUBLAS_DIRS})
#message(STATUS "Cuda include directories" ${CUDA_INCLUDE_DIRS})

#set(CUDA_DIR $ENV{CUDADIR})
#set(CUBLAS_LIBRARIES "-L${CUDA_DIR}/lib64/ -L${CUDA_DIR}/lib64/")

#target_link_libraries(mptc ${CUDA_LIBRARIES} -lcuda -lcudart)
#target_link_libraries(prg_sp2_tc ${CUBLAS_LIBRARIES} ${BLAS_LIBRARIES} ${CUDA_LIBRARIES} -lcublas -lcudadevrt -lcudart)

#set(TC_INCLUDEDIR ${PROJECT_SOURCE_DIR}/install/include)
#set(TC_INSTALLDIR ${PROJECT_SOURCE_DIR}/install/lib64)

#install(TARGETS prg_sp2_tc
#	OBJECTS ARCHIVE DESTINATION ${TC_INSTALLDIR})
#install(TARGETS prg_sp2_tc_fortran
#	OBJECTS ARCHIVE DESTINATION ${TC_INSTALLDIR})

#install(FILES
#        ${PROJECT_BINARY_DIR}/src/sp2_tensorcore/prg_sp2_tensorcore_mod.mod
#        ${SP2_TC_O}
#	${TCSP2_INCLUDE}/tcore_hp_emulator.cuh
#	${TCSP2_INCLUDE}/linalg_tools.cuh
#	${TCSP2_INCLUDE}/prg_sp2_tensorcore.cuh
#	DESTINATION ${TC_INCLUDEDIR})


