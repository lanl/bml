!> \copyright Los Alamos National Laboratory 2015

!> Matrix trace.
module bml_trace_dense_m

  implicit none

  private

  !> Matrix trace.
  interface bml_trace_dense
     module procedure trace_dense_single
     module procedure trace_dense_double
  end interface bml_trace_dense

  public :: bml_trace_dense

contains

  !> Calculate the trace of a matrix.
  !!
  !! \f$ \leftarrow \mathrm{Tr} A \f$
  !!
  !! \param a The matrix.
  function trace_dense_single(a) result(tr_a)

    use bml_type_dense_m

    type(bml_matrix_dense_single_t), intent(in) :: a
    double precision :: tr_a

    integer :: i

    tr_a = 0
    do i = 1, a%n
       tr_a = tr_a+a%matrix(i, i)
    end do

  end function trace_dense_single

  !> Calculate the trace of a matrix.
  !!
  !! \f$ \leftarrow \mathrm{Tr} A \f$
  !!
  !! \param a The matrix.
  function trace_dense_double(a) result(tr_a)

    use bml_type_dense_m

    type(bml_matrix_dense_double_t), intent(in) :: a
    double precision :: tr_a

    integer :: i

    tr_a = 0
    do i = 1, a%n
       tr_a = tr_a+a%matrix(i, i)
    end do

  end function trace_dense_double

end module bml_trace_dense_m
