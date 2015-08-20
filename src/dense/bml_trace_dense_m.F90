!> \copyright Los Alamos National Laboratory 2015

!> Matrix trace.
module bml_trace_dense_m
  implicit none
contains

  !> Calculate the trace of a matrix.
  !!
  !! \f$ \leftarrow \mathrm{Tr} A \f$
  !!
  !! \param A The matrix.
  function trace_dense(A) result(trA)

    use bml_type_dense_m

    type(bml_matrix_dense_t), intent(in) :: A
    double precision :: trA

    integer :: i

    trA = 0
    do i = 1, A%N
       trA = trA+A%matrix(i, i)
    end do

  end function trace_dense

end module bml_trace_dense_m
