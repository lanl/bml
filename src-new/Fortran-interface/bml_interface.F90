!> Interface module.
module bml_interface

  implicit none

  private

  !> The enum values of the C API. Keep this synchronized with the
  !! enum in bml_types.h.
  integer, parameter :: bml_matrix_type_uninitialized_enum_id = 0
  integer, parameter :: bml_matrix_type_dense_enum_id = 1
  integer, parameter :: bml_matrix_precision_single_enum_id = 0
  integer, parameter :: bml_matrix_precision_double_enum_id = 1

  public :: get_enum_id

contains

  !> Convert the matrix type and precisions strings into enum values.
  !!
  !! @param type_string The string used in the Fortran API to identify
  !! the matrix type and precision.
  !! @return The corresponding integer value matching the enum values
  !! in bml_matrix_types_t and bml_matrix_precision_t.
  function get_enum_id(type_string) result(id)

    use bml_types

    character(len=*), intent(in) :: type_string
    integer :: id

    select case(type_string)
    case(BML_PRECISION_SINGLE)
       id = bml_matrix_precision_single_enum_id
    case(BML_PRECISION_DOUBLE)
       id = bml_matrix_precision_double_enum_id
    case(BML_MATRIX_DENSE)
       id = bml_matrix_type_dense_enum_id
    case default
       print *, "unknown type string "//trim(type_string)
       error stop
    end select

  end function get_enum_id

end module bml_interface
