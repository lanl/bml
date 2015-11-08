!> \copyright Los Alamos National Laboratory 2015

!> The cuda-csr matrix types.
module bml_type_cuda_csr_m

  use bml_type_m

  implicit none

  private

  !> The number of columns stored.
  integer, public, parameter :: ELLPACK_M = 10

  !> The cuda-csr matrix type.
  type, abstract, public, extends(bml_matrix_t) :: bml_matrix_cuda_csr_t
     !> The number of entries per row.
     integer, allocatable :: number_entries(:)
     !> Column indices.
     integer, allocatable :: column_index(:, :)
  end type bml_matrix_cuda_csr_t

  !> The bml cuda-csr double precision matrix type.
  type, public, extends(bml_matrix_cuda_csr_t) :: bml_matrix_cuda_csr_double_t
     !> Non-zero matrix elements.
     double precision, allocatable :: matrix(:, :)
   contains
     procedure, nopass :: get_type => get_type_cuda_csr_double
  end type bml_matrix_cuda_csr_double_t

  !> The bml cuda-csr single precision matrix type.
  type, public, extends(bml_matrix_cuda_csr_t) :: bml_matrix_cuda_csr_single_t
     !> Non-zero matrix elements.
     real, allocatable :: matrix(:, :)
   contains
     procedure, nopass :: get_type => get_type_cuda_csr_single
  end type bml_matrix_cuda_csr_single_t

contains

  function get_type_cuda_csr_double() result(type_name)
    character(len=:), allocatable :: type_name
    type_name = "cuda-csr:double"
  end function get_type_cuda_csr_double

  function get_type_cuda_csr_single() result(type_name)
    character(len=:), allocatable :: type_name
    type_name = "cuda-csr:single"
  end function get_type_cuda_csr_single

end module bml_type_cuda_csr_m
