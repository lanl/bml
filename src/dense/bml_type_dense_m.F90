!> \copyright Los Alamos National Laboratory 2015

!> The matrix types.
module bml_type_dense_m

  use bml_type_m

  implicit none

  private

  !> The dense matrix type.
  type, abstract, public, extends(bml_matrix_t) :: bml_matrix_dense_t
  end type bml_matrix_dense_t

  !> The bml dense matrix type.
  type, public, extends(bml_matrix_dense_t) :: bml_matrix_dense_double_t
     !> The dense matrix.
     double precision, allocatable :: matrix(:, :)
   contains
     procedure, nopass :: get_type => get_type_dense_double
  end type bml_matrix_dense_double_t

  !> The bml dense matrix type.
  type, public, extends(bml_matrix_dense_t) :: bml_matrix_dense_single_t
     !> The dense matrix.
     real, allocatable :: matrix(:, :)
   contains
     procedure, nopass :: get_type => get_type_dense_single
  end type bml_matrix_dense_single_t

contains

  function get_type_dense_double() result(type_name)
    character(len=:), allocatable :: type_name
    type_name = "dense:double"
  end function get_type_dense_double

  function get_type_dense_single() result(type_name)
    character(len=:), allocatable :: type_name
    type_name = "dense:single"
  end function get_type_dense_single

end module bml_type_dense_m
