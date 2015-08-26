!> \copyright Los Alamos National Laboratory 2015

!> The ELLPACK matrix types.
module bml_type_ellpack_m

  use bml_type_m

  implicit none

  !> The number of columns stored.
  integer, parameter :: ELLPACK_M = 10

  !> The dense matrix type.
  type, extends(bml_matrix_t) :: bml_matrix_ellpack_t
     !> The number of entries per row.
     integer, allocatable :: number_entries(:)
     !> Column indices.
     integer, allocatable :: column_index(:, :)
  end type bml_matrix_ellpack_t

  !> The bml dense matrix type.
  type, extends(bml_matrix_ellpack_t) :: bml_matrix_ellpack_double_t
     !> Non-zero matrix elements.
     double precision, allocatable :: matrix(:, :)
  end type bml_matrix_ellpack_double_t

  !> The bml dense matrix type.
  type, extends(bml_matrix_ellpack_t) :: bml_matrix_ellpack_single_t
     !> Non-zero matrix elements.
     real, allocatable :: matrix(:, :)
  end type bml_matrix_ellpack_single_t

end module bml_type_ellpack_m
