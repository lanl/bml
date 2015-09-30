set(PRECISION_TYPES single_real double_real single_complex double_complex)

set(C_REAL_TYPES float double "float complex" "double complex")
set(F_REAL_TYPES "real(kind(0.0))" "real(kind(0.0d0))" "complex(kind(0.0))" "complex(kind(0.0d0))")

list(LENGTH PRECISION_TYPES NUMBER_TYPES)
math(EXPR NUMBER_TYPES_MAX "${NUMBER_TYPES}-1")

macro(get_fortran_types INDEX PRECISION_TYPE REAL_TYPE)
  list(GET PRECISION_TYPES ${INDEX} ${PRECISION_TYPE})
  list(GET F_REAL_TYPES ${INDEX} ${REAL_TYPE})
endmacro()

macro(get_c_types INDEX PRECISION_TYPE REAL_TYPE)
  list(GET PRECISION_TYPES ${INDEX} ${PRECISION_TYPE})
  list(GET C_REAL_TYPES ${INDEX} ${REAL_TYPE})
endmacro()
