!> Introspection procedures.
module bml_introspection_m

  implicit none

  private

  interface

     !> Return the matrix size.
     function bml_get_N_C(a) bind(C, name="bml_get_N")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT) :: bml_get_N_C
     end function bml_get_N_C

     function bml_get_row_bandwidth_C(a, i) bind(C, name="bml_get_row_bandwidth")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       integer(C_INT), value, intent(in) :: i
       integer(C_INT) :: bml_get_row_bandwidth_C
     end function bml_get_row_bandwidth_C

  end interface

  public :: bml_get_N
  public :: bml_get_row_bandwidth

contains

  !> Return the matrix size.
  !!
  !!\param a The matrix.
  !!\return The matrix size.
  function bml_get_n(a)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    integer :: bml_get_n

    bml_get_n = bml_get_N_C(a%ptr)

  end function bml_get_n

  !> Get the bandwidth of non-zero elements in a given row.
  !!
  !! @param a The matrix.
  !! @param i The row.
  !! @returns The bandwidth of non-zero elements (bandwidth) on that row.
  function bml_get_row_bandwidth(a, i)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    integer, intent(in) :: i
    integer :: bml_get_row_bandwidth

    bml_get_row_bandwidth = bml_get_row_bandwidth_C(a%ptr, i-1)

  end function bml_get_row_bandwidth

end module bml_introspection_m
