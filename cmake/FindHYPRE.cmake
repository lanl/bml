# - Find the HYPRE library
#
# Usage:
#   find_package(HYPRE [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   HYPRE_FOUND               ... true if HYPRE is found on the system
#   HYPRE_LIBRARY_DIRS        ... full path to HYPRE library
#   HYPRE_INCLUDE_DIRS        ... HYPRE include directory
#   HYPRE_LIBRARIES           ... HYPRE libraries
#
# The following variables will be checked by the function
#   HYPRE_USE_STATIC_LIBS     ... if true, only static libraries are found
#   HYPRE_ROOT                ... if set, the libraries are exclusively searched
#                                 under this path

#If environment variable HYPRE_ROOT is specified, it has same effect as HYPRE_ROOT
if( NOT HYPRE_ROOT AND NOT $ENV{HYPRE_ROOT} STREQUAL "" )
    set( HYPRE_ROOT $ENV{HYPRE_ROOT} )
    # set library directories
    set(HYPRE_LIBRARY_DIRS ${HYPRE_ROOT}/lib)
    # set include directories
    set(HYPRE_INCLUDE_DIRS ${HYPRE_ROOT}/include)
    # set libraries
    find_library(
        HYPRE_LIBRARIES
        NAMES "HYPRE"
        PATHS ${HYPRE_ROOT}
        PATH_SUFFIXES "lib"
        NO_DEFAULT_PATH
    )
    set(HYPRE_FOUND TRUE)
else()
    set(HYPRE_FOUND FALSE)
endif()

