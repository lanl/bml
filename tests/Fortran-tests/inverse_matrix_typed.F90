module inverse_matrix_typed

  use bml
  use prec
  use bml_inverse_m

  implicit none

  public :: test_inverse_matrix_typed

contains

  function test_inverse_matrix_typed(matrix_type, element_kind, element_precision, n, m) &
       & result(test_result)

    character(len=*), intent(in) :: matrix_type, element_kind
    integer, intent(in) :: element_precision
    integer, intent(in) :: n, m
    logical :: test_result
    real(dp) :: threshold

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: a_inverse
    type(bml_matrix_t) :: aux

    threshold = 0.0_MP
    test_result = .false.

    call bml_random_matrix(matrix_type, element_kind, element_precision, n, m, &
         & a)
    call bml_zero_matrix(matrix_type, element_kind, element_precision, n, m, &
         & a_inverse)
    call bml_zero_matrix(matrix_type, element_kind, element_precision, n, m, &
         & aux)
    call bml_print_matrix("A", a, 1, n, 1, n)
    call bml_inverse(a, a_inverse)
    call bml_print_matrix("A_inverse", a_inverse, 1, n, 1, n)

    !! \todo Fixme: multiply is not respecting the correct prec in the scalars
    !! nor in threshold. These have to be set all double presition regrardless
    !! of the presicion of the bml matrix
    call bml_multiply(a, a_inverse, aux, 1.0D0, 0.0D0, threshold)
    call bml_print_matrix("A*A_inverse", aux, 1, n, 1, n)
    test_result = .true.

    call bml_deallocate(a)
    call bml_deallocate(a_inverse)
    call bml_deallocate(aux)

  end function test_inverse_matrix_typed

end module inverse_matrix_typed
