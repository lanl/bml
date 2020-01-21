module scale_matrix_typed

  use bml
  use prec
  use bml_scale_m

  implicit none

  public :: test_scale_matrix_typed

contains

  function test_scale_matrix_typed(matrix_type, element_kind, element_precision, n, m) &
       & result(test_result)

    character(len=*), intent(in) :: matrix_type, element_kind
    integer, intent(in) :: element_precision
    integer, intent(in) :: n, m
    logical :: test_result

    real(DUMMY_PREC), parameter :: alpha = 1.2_MP
    real(DUMMY_PREC) :: abs_tol

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: c

    DUMMY_KIND(DUMMY_PREC), allocatable :: a_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: c_dense(:, :)

    if(element_precision == sp)then
      abs_tol = 1e-6
    elseif(element_precision == dp)then
      abs_tol = 1d-12
    endif

    call bml_random_matrix(matrix_type, element_kind, element_precision, n, m, &
         & a)
    call bml_zero_matrix(matrix_type, element_kind, element_precision, n, m, c)
    call bml_scale(alpha, a, c)

    call bml_export_to_dense(a, a_dense)
    call bml_export_to_dense(c, c_dense)

    if(maxval(abs(alpha * a_dense - c_dense)) > abs_tol) then
      test_result = .false.
      call bml_print_matrix("A", alpha * a_dense, 1, n, 1, n)
      call bml_print_matrix("C", c_dense, 1, n, 1, n)
      print *, "maxval abs difference = ", maxval(abs(alpha * a_dense - c_dense))
      print *, "abs_tol = ", abs_tol
      print *, "matrix element mismatch"
    else
      test_result = .true.
    endif

    call bml_deallocate(a)
    call bml_deallocate(c)

    deallocate(a_dense)
    deallocate(c_dense)

  end function test_scale_matrix_typed

end module scale_matrix_typed
