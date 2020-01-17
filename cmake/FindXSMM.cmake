# - Find the XSMM library
#
# Usage:
#   find_package(XSMM [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   XSMM_FOUND               ... true if magma is found on the system
#   XSMM_LIBRARY_DIRS        ... full path to magma library
#   XSMM_INCLUDE_DIRS        ... magma include directory
#   XSMM_LIBRARIES           ... magma libraries
#
# The following variables will be checked by the function
#   XSMM_USE_STATIC_LIBS     ... if true, only static libraries are found
#   XSMM_ROOT                ... if set, the libraries are exclusively searched
#                                 under this path

#If environment variable XSMM_ROOT is specified, it has same effect as XSMM_ROOT
if( NOT XSMM_ROOT AND NOT $ENV{XSMM_ROOT} STREQUAL "" )
    set( XSMM_ROOT $ENV{XSMM_ROOT} )
    # set library directories
    set(XSMM_LIBRARY_DIRS ${XSMM_ROOT}/lib)
    # set include directories
    set(XSMM_INCLUDE_DIRS ${XSMM_ROOT}/include)
    # set libraries
    find_library(
        XSMM_LIBRARIES
        NAMES "libxsmm"
        PATHS ${XSMM_ROOT}
        PATH_SUFFIXES "lib"
        NO_DEFAULT_PATH
    )
    set(XSMM_FOUND TRUE)
else()
    set(XSMM_FOUND FALSE)
endif()

