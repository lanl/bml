!> \copyright Los Alamos National Laboratory 2015

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

    use bml_type_m
    use bml_type_dense_m
    use bml_trace_dense_m
    use bml_error_m

    class(bml_matrix_t), intent(in) :: a
    double precision :: tr_a

    tr_a = 0

    select type(a)
    type is(bml_matrix_dense_single_t)
       tr_a = bml_trace_dense(a)
    type is(bml_matrix_dense_double_t)
       tr_a = bml_trace_dense(a)
    class default
       call bml_error(__FILE__, __LINE__, "unknown matrix type")
    end select

  end function bml_trace

end module bml_trace_m
