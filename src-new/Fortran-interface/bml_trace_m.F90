!> Matrix trace.
module bml_trace_m
  implicit none
contains

  !> Calculate the trace of a matrix.
  !!
  !! \f$ \leftarrow \mathrm{Tr} \left[ A \right] \f$
  !!
  !! \param a The matrix.
  function bml_trace(a) result(tr_a)

    use bml_types

    class(bml_matrix_t), intent(in) :: a
    double precision :: tr_a

  end function bml_trace

end module bml_trace_m
