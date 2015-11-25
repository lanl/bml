!> Interface module.
module bml_interface_m
  use iso_c_binding
  use bml_types_m
  implicit none

  private

  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  !!
  !! Matrix type is unitialized.
  integer, parameter :: bml_matrix_type_uninitialized_enum_id = 0

  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  !!
  !! Matrix type is dense.
  integer, parameter :: bml_matrix_type_dense_enum_id = 1

  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  !!
  !! Matrix type is ellpack.
  integer, parameter :: bml_matrix_type_ellpack_enum_id = 2

  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  !!
  !! Matrix precision is unitialized.
  integer, parameter :: bml_matrix_precision_uninitialized_id = 0

  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  !!
  !! Matrix precision is single real.
  integer, parameter :: bml_matrix_precision_single_real_enum_id = 1
  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  !!
  !! Matrix precision is double real.
  integer, parameter :: bml_matrix_precision_double_real_enum_id = 2
  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  !!
  !! Matrix precision is single complex.
  integer, parameter :: bml_matrix_precision_single_complex_enum_id = 3
  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  !!
  !! Matrix precision is double complex.
  integer, parameter :: bml_matrix_precision_double_complex_enum_id = 4

  !> The dense matrix element order.
  integer, parameter :: BML_DENSE_COLUMN_MAJOR = 1

  public :: get_type_id, get_prec_id
  public :: BML_DENSE_COLUMN_MAJOR

contains

  !> Convert the matrix type and precisions strings into enum values.
  !!
  !! @param type_string The string used in the Fortran API to identify
  !! the matrix type and precision.
  !! @return The corresponding integer value matching the enum values
  !! in bml_matrix_types_t and bml_matrix_precision_t.
  function get_type_id(type_string) result(id)

    character(len=*), intent(in) :: type_string
    integer(C_INT) :: id

    select case(type_string)
    case(BML_MATRIX_DENSE)
      id = bml_matrix_type_dense_enum_id
    case(BML_MATRIX_ELLPACK)
       id = bml_matrix_type_ellpack_enum_id
    case default
       print *, "unknown matrix type"//trim(type_string)
       error stop
    end select

  end function get_type_id



  function get_prec_id(kind_int) result(id)
    
    integer, intent(in) :: kind_int
    integer(C_INT) :: id

    ! Can not use select case here, as possibly C_* and C_*_COMPLEX
    ! are mapped to the same kind value.
    if (kind_int == C_FLOAT) then
      id = bml_matrix_precision_single_real_enum_id
    else if (kind_int == C_DOUBLE) then
      id = bml_matrix_precision_double_real_enum_id
    else if (kind_int == C_FLOAT_COMPLEX) then
      id = bml_matrix_precision_single_complex_enum_id
    else if (kind_int == C_DOUBLE_COMPLEX) then
      id = bml_matrix_precision_double_complex_enum_id
    else
      print "(A,1X,I0)", "Unknown kind:", kind_int
    end if

  end function get_prec_id


end module bml_interface_m
