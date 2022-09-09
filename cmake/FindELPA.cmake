# - Find the ELPA library
#
# Usage:
#   find_package(ELPA [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   ELPA_FOUND               ... true if elpa is found on the system
#   ELPA_LIBRARY_DIRS        ... full path to elpa library
#   ELPA_INCLUDE_DIRS        ... elpa include directory
#   ELPA_LIBRARIES           ... elpa libraries


find_package(PkgConfig REQUIRED)
pkg_check_modules(ELPA REQUIRED elpa IMPORTED_TARGET)

