!> \copyright Los Alamos National Laboratory 2015

!> The ELLPACK matrix types.
!!
!! The ELLPACK data type was first used in @ref ellpack.
module bml_type_ellpack_m

  use bml_type_m

  implicit none

  private

  !> The number of columns stored.
  integer, public, parameter :: ELLPACK_M = 10

  !> The dense matrix type.
  type, abstract, public, extends(bml_matrix_t) :: bml_matrix_ellpack_t
     !> The number of entries per row.
     integer, allocatable :: number_entries(:)
     !> Column indices.
     integer, allocatable :: column_index(:, :)
     !> The maximum bandwidth.
     integer :: max_bandwidth = -1
  end type bml_matrix_ellpack_t

  !> The bml dense matrix type.
  type, public, extends(bml_matrix_ellpack_t) :: bml_matrix_ellpack_double_t
     !> Non-zero matrix elements.
     double precision, allocatable :: matrix(:, :)
   contains
     procedure, nopass :: get_type => get_type_ellpack_double
  end type bml_matrix_ellpack_double_t

  !> The bml dense matrix type.
  type, public, extends(bml_matrix_ellpack_t) :: bml_matrix_ellpack_single_t
     !> Non-zero matrix elements.
     real, allocatable :: matrix(:, :)
   contains
     procedure, nopass :: get_type => get_type_ellpack_single
  end type bml_matrix_ellpack_single_t

contains

  function get_type_ellpack_double() result(type_name)
    character(len=:), allocatable :: type_name
    type_name = "ellpack:double"
  end function get_type_ellpack_double

  function get_type_ellpack_single() result(type_name)
    character(len=:), allocatable :: type_name
    type_name = "ellpack:single"
  end function get_type_ellpack_single

end module bml_type_ellpack_m
