!> \copyright Los Alamos National Laboratory 2015

!> The matrix types.
module bml_type_ellpack_m

  use bml_type_m

  implicit none

  !> The dense matrix type.
  type, extends(bml_matrix_t) :: bml_matrix_ellpack_t
  end type bml_matrix_ellpack_t

  !> The bml dense matrix type.
  type, extends(bml_matrix_ellpack_t) :: bml_matrix_ellpack_double_t
  end type bml_matrix_ellpack_double_t

  !> The bml dense matrix type.
  type, extends(bml_matrix_ellpack_t) :: bml_matrix_ellpack_single_t
  end type bml_matrix_ellpack_single_t

end module bml_type_ellpack_m
