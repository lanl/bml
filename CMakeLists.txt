cmake_minimum_required(VERSION 3.10...3.17)

message(STATUS "CMake version ${CMAKE_VERSION}")

set(LANGUAGES C CXX)
set(BML_BUILD_FORTRAN_INTERFACE TRUE CACHE BOOL
  "Build the Fortran API (requires a Fortran compiler)")
if(BML_BUILD_FORTRAN_INTERFACE)
  list(APPEND LANGUAGES Fortran)
endif()

project(bml ${LANGUAGES}) # don't move this line as it changes PROJECT_SOURCE_DIR

# The library version is versioned off the major version. If the API
# changes, the library version should be bumped.
set(PROJECT_VERSION_MAJOR "2")
set(PROJECT_VERSION_MINOR "2")
set(PROJECT_VERSION_PATCH "0")

set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set(PROJECT_DESCRIPTION
  "Basic Matrix Library (bml): "
  "A matrix library for linear algebra operations, "
  "supporting several data structures.")
string(REPLACE ";" " " PROJECT_DESCRIPTION "${PROJECT_DESCRIPTION}")

include(GNUInstallDirs)
set(CMAKE_INSTALL_PKG_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
  message(STATUS "No user specified build type, using default...")
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
if(CMAKE_BUILD_TYPE_UPPER STREQUAL DEBUG)
  add_definitions(-DGLOBAL_DEBUG=BML_LOG_DEBUG)
else()
  add_definitions(-DGLOBAL_DEBUG=BML_LOG_INFO)
endif()

include(CheckCCompilerFlag)
check_c_compiler_flag(-fmacro-prefix-map=/a=/b MACRO_PREFIX_MAP_FLAG_WORKS)
if (MACRO_PREFIX_MAP_FLAG_WORKS)
  set(MACRO_PREFIX_MAP_FLAG -fmacro-prefix-map=${CMAKE_SOURCE_DIR}=.)
endif()

set(GNU_C_FLAGS_DEBUG -O0 -g -Wall -Wimplicit -Werror=implicit
  -Wunused-variable -Wno-unknown-pragmas -save-temps -std=c99
  ${MACRO_PREFIX_MAP_FLAG})
set(GNU_C_FLAGS_RELEASE -O3 -g -std=c99 -DNDEBUG
  ${MACRO_PREFIX_MAP_FLAG})
set(GNU_C_FLAGS_RELWITHDEBINFO -O3 -g -std=c99 -DNDEBUG
  ${MACRO_PREFIX_MAP_FLAG})
set(GNU_Fortran_FLAGS_DEBUG -O0 -g -fcheck=all
  -ffree-line-length-none)
set(GNU_Fortran_FLAGS_RELEASE -O3 -g -ffree-line-length-none)
set(GNU_Fortran_FLAGS_RELWITHDEBINFO -O3 -g -ffree-line-length-none
  -DNDEBUG)

set(Intel_C_FLAGS_DEBUG -O0 -g -std=c99 -check=conversions,stack,uninit
  -traceback -qopt-report=5 -DINTEL_SDK)
set(Intel_C_FLAGS_RELEASE -O2 -g -std=c99 -qopt-report=5 -DNDEBUG -DINTEL_SDK)
set(Intel_C_FLAGS_RELWITHDEBINFO -O2 -g -std=c99 -qopt-report=5 -DNDEBUG -DINTEL_SDK)
set(Intel_Fortran_FLAGS_DEBUG -O0 -g -check all -assume realloc_lhs
  -traceback -qopt-report=5 -DINTEL_SDK)
set(Intel_Fortran_FLAGS_RELEASE -O2 -g -assume realloc_lhs -qopt-report=5 -DINTEL_SDK)
set(Intel_Fortran_FLAGS_RELWITHDEBINFO -O2 -g -assume realloc_lhs
  -qopt-report=5 -DNDEBUG -DINTEL_SDK)

set(XL_C_FLAGS_DEBUG -O0 -g)
set(XL_C_FLAGS_RELEASE -O3 -g -std=c99 -DNDEBUG)
set(XL_C_FLAGS_RELWITHDEBINFO -O2 -g -std=c99 -DNDEBUG)
set(XL_Fortran_FLAGS_DEBUG -O0 -g)
set(XL_Fortran_FLAGS_RELEASE -O3 -g)
set(XL_Fortran_FLAGS_RELWITHDEBINFO -O3 -g -DNDEBUG)

set(Clang_C_FLAGS_DEBUG -O0 -g -save-temps -std=c99)
set(Clang_C_FLAGS_RELEASE -O2 -g -std=c99 -DNDEBUG)
set(Clang_C_FLAGS_RELWITHDEBINFO -O2 -g -std=c99 -DNDEBUG)

# If Cray Fortran compiler then assume Cray C compiler
#  (This is needed as Cray C compiler ID is Clang)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
  set(EXTRA_FFLAGS "-ef -DCRAY_SDK ${EXTRA_FFLAGS}")
  set(EXTRA_CFLAGS "-DCRAY_SDK ${EXTRA_CFLAGS}")
endif()

set(MALLOC_ALIGNMENT 64 CACHE INT "Alignment boundary for memory allocations")
add_definitions(-DMALLOC_ALIGNMENT=${MALLOC_ALIGNMENT})

option(DONT_TOUCH_MY_FLAGS "Don't touch compiler flags" OFF)
if(CMAKE_BUILD_TYPE_UPPER AND NOT DONT_TOUCH_MY_FLAGS)
  if(CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG"
      OR CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE"
      OR CMAKE_BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
    if(CMAKE_C_FLAGS STREQUAL "")
      if(CMAKE_C_COMPILER_ID STREQUAL "GNU"
          OR CMAKE_C_COMPILER_ID STREQUAL "Intel"
          OR CMAKE_C_COMPILER_ID STREQUAL "Clang"
          OR CMAKE_C_COMPILER_ID STREQUAL "Cray"
          OR CMAKE_C_COMPILER_ID STREQUAL "XL")
        set(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}
          ${${CMAKE_C_COMPILER_ID}_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}})
        string(REPLACE ";" " "
          CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}
          "${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
        if(DEFINED EXTRA_CFLAGS)
          set(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}
            "${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} ${EXTRA_CFLAGS}")
        endif()
        message(STATUS "Setting C compiler flags to "
          "${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
      else()
        message(STATUS "Unknown C compiler ${CMAKE_C_COMPILER_ID}")
      endif()
    else()
      message(STATUS "Using user supplied C compiler flags: ${CMAKE_C_FLAGS}")
      set(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}
        ${CMAKE_C_FLAGS})
    endif()

    if(CMAKE_Fortran_FLAGS STREQUAL "")
      if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU"
          OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel"
          OR CMAKE_Fortran_COMPILER_ID STREQUAL "Clang"
          OR CMAKE_Fortran_COMPILER_ID STREQUAL "Cray"
          OR CMAKE_Fortran_COMPILER_ID STREQUAL "XL")
        set(CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}
          ${${CMAKE_Fortran_COMPILER_ID}_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}})
        string(REPLACE ";" " "
          CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}
          "${CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
        if(DEFINED EXTRA_FFLAGS)
          set(CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}
            "${CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} ${EXTRA_FFLAGS}")
        endif()
        message(STATUS "Setting Fortran compiler flags to "
          "${CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
      else()
        message(STATUS "Unknown Fortran compiler ${CMAKE_Fortran_COMPILER_ID}")
      endif()
    else()
      message(STATUS "Using user supplied Fortran compiler flags: ${CMAKE_Fortran_FLAGS}")
      set(CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}
      ${CMAKE_Fortran_FLAGS})
    endif()
  endif()
endif()

set(BML_LINK_FLAGS "" CACHE STRING "Additional linker flags")
list(APPEND LINK_LIBRARIES ${BML_LINK_FLAGS})

include(FindThreads)
list(APPEND LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

set(BML_OPENMP TRUE CACHE BOOL "Whether to compile with OpenMP support")
if(BML_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
    if(CMAKE_VERSION VERSION_LESS 3.1)
      # There is not Fortran support in FindOpenMP before version 3.1. We will
      # have to hack this.
      message(WARNING "Versions of <cmake-3.1 can not detect Fortran "
        "compiler OpenMP flags. We will assume that the flags found for "
        "the C compiler also work for the Fortran compiler. If this is "
        "not the case, please send email to <nbock@lanl.gov>.")
      set(OpenMP_Fortran_FLAGS ${OpenMP_C_FLAGS})
    endif()
  else()
    message(WARNING "Could not get the compilers to use OpenMP. "
      "Will pretend that this never happened and compile the library "
      "without OpenMP.")
    set(OpenMP_C_FLAGS "")
    set(OpenMP_CXX_FLAGS "")
    set(OpenMP_Fortran_FLAGS "")
  endif()
else()
  message(STATUS "Will not build with OpenMP")
endif()

if(MKL_GPU)
	add_definitions(-DMKL_GPU)
endif()

set(BML_MPI FALSE CACHE BOOL "Compile with MPI support")
if(BML_MPI)
  message(STATUS "Will build with MPI")
  find_package(MPI REQUIRED)
  #possibly override MPIEXEC_EXECUTABLE,...
  if(NOT ${BML_MPIEXEC_EXECUTABLE} STREQUAL "")
    set(MPIEXEC_EXECUTABLE ${BML_MPIEXEC_EXECUTABLE})
  endif()
  if(BML_MPIEXEC_NUMPROCS_FLAG)
    message(STATUS "override MPIEXEC_NUMPROCS_FLAG with ${BML_MPIEXEC_NUMPROCS_FLAG}")
    set(MPIEXEC_NUMPROCS_FLAG ${BML_MPIEXEC_NUMPROCS_FLAG})
  else()
    set(MPIEXEC_NUMPROCS_FLAG "-n")
  endif()
  if(BML_MPIEXEC_NUMPROCS)
    message(STATUS "override MPIEXEC_NUMPROCS with ${BML_MPIEXEC_NUMPROCS}")
    set(MPIEXEC_NUMPROCS ${BML_MPIEXEC_NUMPROCS})
  else()
    set(MPIEXEC_NUMPROCS "4")
  endif()
  if(BML_MPIEXEC_PREFLAGS)
    message(STATUS "override MPIEXEC_PREFLAGS with ${BML_MPIEXEC_PREFLAGS}")
    set(MPIEXEC_PREFLAGS ${BML_MPIEXEC_PREFLAGS})
  endif()
  message(STATUS "MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
  message(STATUS "MPIEXEC_NUMPROCS_FLAG: ${MPIEXEC_NUMPROCS_FLAG}")
  message(STATUS "MPIEXEC_PREFLAGS: ${MPIEXEC_PREFLAGS}")
  message(STATUS "BML_MPIEXEC_NUMPROCS_FLAG: ${BML_MPIEXEC_NUMPROCS_FLAG}")
  add_definitions(-DBML_USE_MPI)
endif()

set(BML_CUSPARSE FALSE CACHE BOOL "Whether to compile with CuSPARSE support")
set(BML_ROCSPARSE FALSE CACHE BOOL "Whether to compile with rocSPARSE support")
set(BML_CUDA FALSE CACHE BOOL "Whether to compile with CUDA support")
set(BML_CUSOLVER FALSE CACHE BOOL "Whether to compile with cuSolver support")
set(BML_MAGMA FALSE CACHE BOOL "Whether to use MAGMA library")
set(BML_OMP_OFFLOAD OFF CACHE BOOL "Compile with OpenMP GPU Offload support")
set(BML_OFFLOAD_ARCH "NVIDIA" CACHE STRING "Offload architecture")
set_property(CACHE BML_OFFLOAD_ARCH PROPERTY STRINGS "NVIDIA" "AMD")

if(BML_OMP_OFFLOAD AND NOT BML_OPENMP)
  message(FATAL_ERROR "BML_OMP_OFFLOAD requires BML_OPENMP")
endif()

if(BML_CUDA OR BML_CUSOLVER OR BML_CUSPARSE OR (BML_OMP_OFFLOAD AND BML_OFFLOAD_ARCH STREQUAL "NVIDIA"))
  if(${CMAKE_VERSION} VERSION_LESS 3.17)
    message(FATAL_ERROR "CUDA/cuSparse/cuSolver support requires cmake3.17")
  endif()
  include(FindCUDAToolkit)
  if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDAToolkit: ${CUDAToolkit_TARGET_DIR}")
  endif()
endif()

if(BML_ROCSOLVER OR BML_ROCSPARSE OR BML_MAGMA OR (BML_OMP_OFFLOAD AND BML_OFFLOAD_ARCH STREQUAL "AMD"))
  find_package(hip QUIET)
endif()

if(BML_OMP_OFFLOAD)
  message(STATUS "Will build with OpenMP GPU Offload support")

  add_definitions(-DUSE_OMP_OFFLOAD)

  if(OpenMP_C_VERSION LESS 4.5)
    message(FATAL_ERROR "OpenMP offload requires OpenMP version >= 4.5")
  endif()

  include(CheckCXXCompilerFlag)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(BML_OFFLOAD_ARCH STREQUAL "NVIDIA")
      if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "Offload to NVIDIA selected but CUDA not found.")
      endif()
      set(OpenMP_CXX_OFFLOAD_FLAG "-foffload=-misa=sm_35 -foffload=nvptx-none -foffload='-lm'")
    endif()
  elseif(MAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(OpenMP_CXX_OFFLOAD_FLAG "-foffload='-lm'")
  elseif(MAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(BML_OFFLOAD_ARCH STREQUAL "NVIDIA")
	    set(OpenMP_CXX_OFFLOAD_FLAG "-O2 -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDAToolkit_TARGET_DIR}")
    elseif(BML_OFFLOAD_ARCH STREQUAL "AMD")
      if(NOT hip_FOUND)
        message(FATAL_ERROR "Offload to AMD selected but HIP not found.")
      endif()
      if(DEFINED ENV{CRAYPE_VERSION})
        set(OpenMP_CXX_OFFLOAD_FLAG "")
      else()
	message(FATAL_ERROR "BML_OFFLOAD_ARCH is AMD but Cray compiler not detected")
      endif()
    endif()
  elseif(MAKE_CXX_COMPILER_ID STREQUAL "XL")
    set(OpenMP_CXX_OFFLOAD_FLAG "-qsmp=omp -qoffload")
  endif()
  check_cxx_compiler_flag("${OpenMP_CXX_OFFLOAD_FLAG}" OFFLOAD_CXX_FOUND)
  if(OFFLOAD_CXX_FOUND AND NOT DONT_TOUCH_MY_FLAGS)
    message(STATUS "Found GPU offload flags")
    set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${OpenMP_CXX_OFFLOAD_FLAG}")
    set(OpenMP_C_FLAGS "${OpenMP_C_FLAGS} ${OpenMP_CXX_OFFLOAD_FLAG}")
    set(OpenMP_Fortran_FLAGS "${OpenMP_Fortran_FLAGS} ${OpenMP_CXX_OFFLOAD_FLAG}")
  endif()

  # list(APPEND LANGUAGES CUDA)

  if(BML_OFFLOAD_ARCH EQUAL "NVIDIA")

    set(GPU_ARCH "sm_60" CACHE STRING "LAMMPS GPU CUDA SM architecture")
    set_property(CACHE GPU_ARCH PROPERTY STRINGS sm_50 sm_60 sm_70)

  endif()
endif()

set(BML_USE_DEVICE (BML_CUDA OR BML_MAGMA OR BML_CUSOLVER OR BML_ROCSOLVER OR BML_CUSPARSE OR BML_ROCSPARSE))
if(BML_USE_DEVICE)
  message(STATUS "Will use device libraries for some solvers")
  if (CUDAToolkit_FOUND)
    message(STATUS "CUDA CUBLAS libraries: ${CUDA_cublas_LIBRARY}")
    list(APPEND LINK_LIBRARIES ${CUDA_cublas_LIBRARY})
    list(APPEND LINK_LIBRARIES ${CUDA_cudart_LIBRARY})
    include_directories(${CUDAToolkit_INCLUDE_DIR})
    if(BML_CUSOLVER)
      message(STATUS "Will use cuSOLVER")
      message(STATUS "CUDA cuSOLVER library: ${CUDA_cusolver_LIBRARY}")
    endif()
    if(BML_CUSPARSE)
      message(STATUS "Will use cuSPARSE")
      add_definitions(-DBML_USE_CUSPARSE)
      message(STATUS "CUDA cuSPARSE library: ${CUDA_cusparse_LIBRARY}")
      list(APPEND LINK_LIBRARIES ${CUDA_cusparse_LIBRARY})      
    endif()
  elseif(hip_FOUND)
    message(STATUS "HIP libraries: ${hip_LIBRARIES}")
    include_directories(${hip_INCLUDE_DIRS})
    if(BML_MAGMA)
      find_package(rocblas REQUIRED)
      message(STATUS "HIP rocBLAS libraries: ${rocblas_LIBRARIES}")
      list(APPEND LINK_LIBRARIES ${rocblas_LIBRARIES})
      include_directories(${rocblas_INCLUDE_DIRS})
      find_package(hipblas REQUIRED)
      include_directories(${hipblas_INCLUDE_DIRS})
      find_package(hipsparse REQUIRED)
      include_directories(${hipsparse_INCLUDE_DIRS})
      if(BML_ROCSOLVER)
        message(STATUS "Will use rocSOLVER")
        find_package(rocsolver REQUIRED)
        message(STATUS "HIP rocSOLVER libraries: ${rocsolver_LIBRARIES}")
        include_directories(${rocsolver_INCLUDE_DIRS})
        list(APPEND LINK_LIBRARIES ${rocsolver_LIBRARIES})
      endif()
    endif()
    if(BML_ROCSPARSE)
      message(STATUS "Will use rocSPARSE")
      add_definitions(-DBML_USE_ROCSPARSE)
      if(BML_OFFLOAD_ARCH STREQUAL "AMD")
        add_definitions(-D__HIP_PLATFORM_AMD__)
      elseif(BML_OFFLOAD_ARCH STREQUAL "NVIDIA")
        add_definitions(-D__HIP_PLATFORM_NVIDIA__)
      endif()
      find_package(rocsparse REQUIRED)
      message(STATUS "HIP rocSPARSE libraries: ${rocsparse_LIBRARIES}")
      include_directories(${rocsparse_INCLUDE_DIRS})
      list(APPEND LINK_LIBRARIES ${rocsparse_LIBRARIES})
    endif()      
    list(APPEND LINK_LIBRARIES ${hip_LIBRARIES})
  endif()
endif()

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")


if(BML_MAGMA)
  message(STATUS "Search for MAGMA...")
  find_package(MAGMA)

  if (MAGMA_FOUND)
    add_definitions(-DBML_USE_MAGMA)
    message(STATUS
      "MAGMA was found:\n"
      "  MAGMA_INCLUDE_DIRS: ${MAGMA_INCLUDE_DIRS}\n"
      "  MAGMA_LIBRARY_DIRS: ${MAGMA_LIBRARY_DIRS}\n"
      "  MAGMA_LIBRARIES: ${MAGMA_LIBRARIES}"
    )
    include_directories(${MAGMA_INCLUDE_DIRS})
    if(BML_CUSOLVER)
      add_definitions(-DBML_USE_CUSOLVER)
    endif()
    if(BML_ROCSOLVER)
      add_definitions(-DBML_USE_ROCSOLVER)
    endif()
  else()
    message(FATAL_ERROR "MAGMA was not found")
  endif()
endif()

set(BML_ELPA FALSE CACHE BOOL "Whether to use ELPA library")
if(BML_ELPA)
  message(STATUS "Search for ELPA in directory ${ELPA_DIR}\n")
  find_package(ELPA REQUIRED)

  if(${ELPA_FOUND})
    message(STATUS "ELPA was found:\n"
      "  ELPA_INCLUDE_DIRS: ${ELPA_INCLUDE_DIRS}\n"
      "  ELPA_LIBRARY_DIRS: ${ELPA_LIBRARY_DIRS}\n"
      "  ELPA_LIBRARIES: ${ELPA_LIBRARIES}"
    )
    add_definitions(-DBML_USE_ELPA)
    include_directories(${ELPA_INCLUDE_DIRS})
    link_directories(${CUDAToolkit_LIBRARY_DIR})
    link_directories(${ELPA_LIBRARY_DIRS})
    list(APPEND LINK_LIBRARIES ${ELPA_LIBRARIES})
  endif()
endif()

set(BML_SCALAPACK FALSE CACHE BOOL "Whether to use ScaLAPACK library")
if(BML_SCALAPACK)
  add_definitions(-DBML_USE_SCALAPACK)
  message(STATUS "SCALAPACK_LIBRARIES=${SCALAPACK_LIBRARIES}")
endif()

set(BML_INTERNAL_BLAS FALSE
  CACHE BOOL "Use internal BLAS functions")

set(BML_INTERNAL_GEMM ${BML_INTERNAL_BLAS}
  CACHE BOOL "Use internal version of ?gemm BLAS functions")
if(BML_INTERNAL_GEMM)
  message(STATUS "Will use internal gemm")
  add_definitions(-DBML_INTERNAL_GEMM)
endif()

set(BLAS_VENDOR ""
  CACHE STRING "If set, the preferred BLAS/LAPACK vendor. Possible choices: {OpenBLAS,Intel,ACML,IBMESSL,...}")

if(BLAS_VENDOR STREQUAL "None")
  set(NOBLAS TRUE)
  add_definitions(-DNOBLAS)
elseif(NOT BLAS_LIBRARIES)
  set(BLA_VENDOR ${BLAS_VENDOR})
  message(STATUS "BLA_VENDOR=${BLA_VENDOR}")
  include(FindBLAS)
  if(NOT BLAS_FOUND)
    message(FATAL_ERROR "Can not find suitable BLAS library")
  endif()
endif()
message(STATUS "BLAS_LIBRARIES=${BLAS_LIBRARIES}")
if(BLAS_LIBRARIES)
  set(BLAS_FOUND TRUE)
endif()

if(LAPACK_LIBRARIES)
  set(LAPACK_FOUND TRUE)
endif()

if(NOT (LAPACK_FOUND OR NOBLAS OR MAGMA_FOUND))
  include(FindLAPACK)
  if(NOT LAPACK_FOUND)
    message(FATAL_ERROR "Can not find suitable LAPACK library")
  endif()
endif()
message(STATUS "LAPACK_LIBRARIES=${LAPACK_LIBRARIES}")

#check existence of required math and linear algebra functions
#in third party libraries
include(CheckFunctionExists)
include(CheckFortranFunctionExists)
include(${PROJECT_SOURCE_DIR}/cmake/bmlCheckCFortranFunctionExists.cmake)

if(BLAS_FOUND)
  add_definitions(-DHAVE_BLAS)

  set(CMAKE_REQUIRED_LIBRARIES
    ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})
  bml_check_C_Fortran_function_exists(dgemm C_DGEMM REQUIRED)
  bml_check_C_Fortran_function_exists(sgemm C_SGEMM REQUIRED)
  bml_check_C_Fortran_function_exists(cgemm C_CGEMM REQUIRED)
  bml_check_C_Fortran_function_exists(zgemm C_ZGEMM REQUIRED)
  bml_check_C_Fortran_function_exists(sscal C_SSCAL REQUIRED)
  bml_check_C_Fortran_function_exists(dscal C_DSCAL REQUIRED)
  bml_check_C_Fortran_function_exists(cscal C_CSCAL REQUIRED)
  bml_check_C_Fortran_function_exists(zscal C_ZSCAL REQUIRED)
  bml_check_C_Fortran_function_exists(saxpy C_SAXPY REQUIRED)
  bml_check_C_Fortran_function_exists(daxpy C_DAXPY REQUIRED)
  bml_check_C_Fortran_function_exists(caxpy C_CAXPY REQUIRED)
  bml_check_C_Fortran_function_exists(zaxpy C_ZAXPY REQUIRED)
  add_definitions(
    -DC_SGEMM=${C_SGEMM}
    -DC_DGEMM=${C_DGEMM}
    -DC_CGEMM=${C_CGEMM}
    -DC_ZGEMM=${C_ZGEMM}
    -DC_SSCAL=${C_SSCAL}
    -DC_DSCAL=${C_DSCAL}
    -DC_CSCAL=${C_CSCAL}
    -DC_ZSCAL=${C_ZSCAL}
    -DC_SAXPY=${C_SAXPY}
    -DC_DAXPY=${C_DAXPY}
    -DC_CAXPY=${C_CAXPY}
    -DC_ZAXPY=${C_ZAXPY})
endif()

#check for functions in Lapack
#(or OpenBLAS since OpenBLAS typically contains Lapack)
if(LAPACK_FOUND OR (BLAS_VENDOR STREQUAL "OpenBLAS"))
  add_definitions(-DHAVE_LAPACK)

  set(CMAKE_REQUIRED_LIBRARIES
    ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})
  bml_check_C_Fortran_function_exists(ssyev C_SSYEV REQUIRED)
  bml_check_C_Fortran_function_exists(dsyev C_DSYEV REQUIRED)
  bml_check_C_Fortran_function_exists(cheevr C_CHEEVR REQUIRED)
  bml_check_C_Fortran_function_exists(zheevr C_ZHEEVR REQUIRED)
  bml_check_C_Fortran_function_exists(sgetrf C_SGETRF REQUIRED)
  bml_check_C_Fortran_function_exists(dgetrf C_DGETRF REQUIRED)
  bml_check_C_Fortran_function_exists(cgetrf C_CGETRF REQUIRED)
  bml_check_C_Fortran_function_exists(zgetrf C_ZGETRF REQUIRED)
  bml_check_C_Fortran_function_exists(sgetri C_SGETRI REQUIRED)
  bml_check_C_Fortran_function_exists(dgetri C_DGETRI REQUIRED)
  bml_check_C_Fortran_function_exists(cgetri C_CGETRI REQUIRED)
  bml_check_C_Fortran_function_exists(zgetri C_ZGETRI REQUIRED)
  bml_check_C_Fortran_function_exists(slacpy C_SLACPY REQUIRED)
  bml_check_C_Fortran_function_exists(dlacpy C_DLACPY REQUIRED)
  bml_check_C_Fortran_function_exists(clacpy C_CLACPY REQUIRED)
  bml_check_C_Fortran_function_exists(zlacpy C_ZLACPY REQUIRED)
  add_definitions(
    -DC_SSYEV=${C_SSYEV}
    -DC_DSYEV=${C_DSYEV}
    -DC_CHEEVR=${C_CHEEVR}
    -DC_ZHEEVR=${C_ZHEEVR}
    -DC_SGETRF=${C_SGETRF}
    -DC_DGETRF=${C_DGETRF}
    -DC_CGETRF=${C_CGETRF}
    -DC_ZGETRF=${C_ZGETRF}
    -DC_SGETRI=${C_SGETRI}
    -DC_DGETRI=${C_DGETRI}
    -DC_CGETRI=${C_CGETRI}
    -DC_ZGETRI=${C_ZGETRI}
    -DC_SLACPY=${C_SLACPY}
    -DC_DLACPY=${C_DLACPY}
    -DC_CLACPY=${C_CLACPY}
    -DC_ZLACPY=${C_ZLACPY})
endif()

if(BML_SCALAPACK)
  set(CMAKE_REQUIRED_LIBRARIES ${SCALAPACK_LIBRARIES}
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${OpenMP_C_FLAGS})
  message(STATUS "SCALAPACK_LIBRARIES=${SCALAPACK_LIBRARIES}")
  bml_check_C_Fortran_function_exists(numroc NUMROC REQUIRED)
  bml_check_C_Fortran_function_exists(pssyevd PSSYEVD REQUIRED)
  bml_check_C_Fortran_function_exists(pdsyevd PDSYEVD REQUIRED)
  bml_check_C_Fortran_function_exists(pcheevd PCHEEVD REQUIRED)
  bml_check_C_Fortran_function_exists(pzheevd PZHEEVD REQUIRED)
  add_definitions(
    -DPSSYEVD=${PSSYEVD}
    -DPDSYEVD=${PDSYEVD}
    -DPCHEEVD=${PCHEEVD}
    -DPZHEEVD=${PZHEEVD}
    -DNUMROC=${NUMROC})
endif()

#append linear algebra libraries to list of libraries to link with
#warning: order matters!
if(MAGMA_FOUND)
  list(APPEND LINK_LIBRARIES ${MAGMA_LIBRARIES})
  if(BML_CUSOLVER)
    list(APPEND LINK_LIBRARIES ${CUDA_cusolver_LIBRARY})
  elseif(BML_ROCSOLVER)
    list(APPEND LINK_LIBRARIES ${rocsolver_LIBRARIES})
  endif()
endif()
if(BML_SCALAPACK)
  list(APPEND LINK_LIBRARIES ${SCALAPACK_LIBRARIES})
endif()
if(LAPACK_FOUND)
  list(APPEND LINK_LIBRARIES ${LAPACK_LIBRARIES})
endif()
if(BLAS_FOUND)
  list(APPEND LINK_LIBRARIES ${BLAS_LIBRARIES})
endif()
list(APPEND LINK_LIBRARIES -lm)

# Check whether the compiler supports complex types. The Nvidia
# compiler is known not to support them for example.

set(BML_COMPLEX TRUE
  CACHE BOOL "Whether to build complex types")

if(BML_COMPLEX)
  try_compile(SUPPORTS_COMPLEX ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake/test_complex.c
    OUTPUT_VARIABLE COMPLEX_OUTPUT
    )
  message(STATUS "Compile output: ${COMPLEX_OUTPUT}")
  if(SUPPORTS_COMPLEX)
    message(STATUS "The compiler supports complex types")
    add_definitions(-DBML_COMPLEX)
  else()
    message(STATUS "The compiler does not support complex types. Will skip them.")
    set(BML_COMPLEX FALSE)
  endif()
else()
  message(STATUS "Complex types disabled by user request")
endif()

set(BML_XSMM FALSE CACHE BOOL "Whether to use XSMM library")
if(NOT (NOBLAS))
  if(BML_XSMM)
    find_package(XSMM)
  endif()
  if(XSMM_FOUND)
    message(STATUS "Use XSMM")
    add_definitions(-DBML_USE_XSMM)
    include_directories(${XSMM_INCLUDE_DIRS})
    list(APPEND LINK_LIBRARIES ${XSMM_LIBRARY_DIRS}/libxsmm.so
                               ${BLAS_LIBRARIES})
  endif()
endif()

if(BML_ELLBLOCK_MEMPOOL)
  message(STATUS "Use memory pool for elllock")
  add_definitions(-DBML_ELLBLOCK_USE_MEMPOOL)
endif()

add_definitions(-D_POSIX_C_SOURCE=200112L)

check_function_exists(posix_memalign HAVE_POSIX_MEMALIGN)
if(HAVE_POSIX_MEMALIGN)
  add_definitions(-DHAVE_POSIX_MEMALIGN)
endif()

set(INTEL_OPT FALSE
  CACHE BOOL "Whether to use Intel compiler specific optimizations")

if(INTEL_OPT)
  if(CMAKE_C_COMPILER_ID STREQUAL "Intel")
    add_definitions(-DINTEL_OPT)
  else()
    message(FATAL_ERROR "Compiler ID must be Intel when INTEL_OPT=yes")
  endif()
endif()

find_program(ADDR2LINE addr2line)
if(ADDR2LINE)
  message(STATUS "Found addr2line, backtrace will resolve line numbers")
  add_definitions(-DHAVE_ADDR2LINE -DADDR2LINE="${ADDR2LINE}")
else()
  message(STATUS "Did not find addr2line")
endif()

set(DIST_FILES CMakeLists.txt)

include(FindDoxygen)

if(DOXYGEN_FOUND)
  set(DOXYGEN_INPUT
    ${PROJECT_SOURCE_DIR}/src/C-interface
    ${PROJECT_SOURCE_DIR}/src/Fortran-interface)
  string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT}")
  configure_file(documentation/Doxyfile.in Doxyfile)
  add_custom_target(docs
    COMMAND rm -rvf ${PROJECT_SOURCE_DIR}/docs/API/*
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile)
else()
  add_custom_target(docs)
endif()

set(BML_TESTING FALSE
  CACHE BOOL "Whether to build the test suite.")
set(BML_VALGRIND FALSE
  CACHE BOOL "Whether to test for memory leaks with valgrind")
set(VALGRIND_COMMON_ARGS --fullpath-after=src/ --fullpath-after=tests/
  --error-exitcode=1 --leak-check=full --show-leak-kinds=all
  --read-var-info=yes --track-origins=yes)

if(BML_TESTING)
  message(STATUS "Setting up test suite")
  find_program(VALGRIND valgrind
    DOC "valgrind - a suite of tools for debugging and profiling programs")
  if(VALGRIND AND BML_VALGRIND)
    execute_process(COMMAND ${VALGRIND} --version
      OUTPUT_VARIABLE VALGRIND_VERSION)
    string(REPLACE "\n" "" VALGRIND_VERSION ${VALGRIND_VERSION})
    message(STATUS "Will test for memory leaks (${VALGRIND_VERSION})")
  else()
    message(STATUS "Will _not_ test for memory leaks")
  endif()
  enable_testing()
  add_subdirectory(tests)
endif()

set(BML_VERSION "${PROJECT_VERSION}")

find_program(GIT git)
if(GIT)
  execute_process(COMMAND ${GIT} describe
    RESULT_VARIABLE IS_GIT_REPOSITORY
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DESCRIBE_OUTPUT
    ERROR_QUIET)
  if(NOT IS_GIT_REPOSITORY EQUAL 0)
    message(WARNING "Not in git repository, disabling dist target")
    add_custom_target(dist)
  else()
    find_program(TAR tar)
    if(TAR)
      message(STATUS "Creating dist target")
      add_custom_target(dist
        COMMAND ${GIT} archive
        --format=tar.gz
        --prefix=bml-${PROJECT_VERSION}/
        --output=${CMAKE_BINARY_DIR}/bml-${PROJECT_VERSION}.tar.gz
        master
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      string(STRIP ${GIT_DESCRIBE_OUTPUT} BML_VERSION)
    else()
      message(WARNING "No tar found, disabling dist target")
      add_custom_target(dist)
    endif()
  endif()
else()
  message(WARNING "No git command found, disabling dist target")
  add_custom_target(dist)
endif()
add_definitions(-DPROJECT_VERSION="${PROJECT_VERSION}")
add_definitions(-DBML_VERSION="${BML_VERSION}")
message(STATUS "Setting BML_VERSION to '${BML_VERSION}'")

include(FindPythonInterp)
if(NOT PYTHONINTERP_FOUND)
  message(FATAL "Could not find Python interpreter")
endif()
configure_file(${PROJECT_SOURCE_DIR}/scripts/convert-template.in
  ${CMAKE_BINARY_DIR}/convert-template)

message(STATUS "When building executables with the bml library, please use the following link flags")

if(OPENMP_FOUND)
  message(STATUS "Additional Fortran compiler link flags: '${OpenMP_Fortran_FLAGS}'")
  message(STATUS "Additional C compiler link flags: '${OpenMP_C_FLAGS}'")
endif()

message(STATUS "LINK_LIBRARIES=${LINK_LIBRARIES}")
string(REPLACE ";" " " LINKLINE "${LINK_LIBRARIES}")
string(STRIP "${LINKLINE}" LINKLINE)
message(STATUS "Link flags: '${LINKLINE}'")

configure_file(src/bml.pc.in bml.pc)
install(FILES ${CMAKE_BINARY_DIR}/bml.pc
  DESTINATION ${CMAKE_INSTALL_PKG_CONFIG_DIR})

add_subdirectory(src)
