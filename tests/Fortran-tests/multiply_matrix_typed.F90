module multiply_matrix_typed

  use bml
  use prec
  use bml_multiply_m

  implicit none

  public :: test_multiply_matrix_typed

contains

  function test_multiply_matrix_typed(matrix_type, element_kind, element_precision, n, m) &
       & result(test_result)

    character(len=*), intent(in) :: matrix_type, element_kind
    integer, intent(in) :: element_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b
    type(bml_matrix_t) :: c
    type(bml_matrix_t) :: d
    type(bml_matrix_t) :: f
    type(bml_matrix_t) :: g


    DUMMY_KIND(DUMMY_PREC), allocatable :: a_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: b_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: c_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: d_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: e_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: f_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: g_dense(:, :)

    real(dp), allocatable :: trace(:)

    real(dp) :: alpha = -0.8_dp
    real(dp) :: beta = 1.2_dp
    real(dp) :: threshold = 0.0_dp
    real(dp) :: ONE = 1.0_dp
    real(dp) :: ZERO = 0.0_dp
    real(DUMMY_PREC) :: abs_tol

    if(element_precision == sp)then
      abs_tol = 1e-6
    elseif(element_precision == dp)then
      abs_tol = 1d-12
    endif

    call bml_random_matrix(matrix_type, element_kind, element_precision, n, m, &
         & a)
    call bml_identity_matrix(matrix_type, element_kind, element_precision, n, &
         & m, b)
    call bml_identity_matrix(matrix_type, element_kind, element_precision, n, &
         & m, c)
    call bml_copy_new(c, d)

    call bml_multiply(a, b, d, alpha, beta, threshold)

    call bml_zero_matrix(matrix_type, element_kind, element_precision, n, m, f)
    call bml_zero_matrix(matrix_type, element_kind, element_precision, n, m, g)
    call bml_multiply_x2(a, f, threshold, trace)
    call bml_multiply(a, a, g, ONE, ZERO, threshold)

    test_result = .true.

    call bml_export_to_dense(a, a_dense)
    call bml_export_to_dense(b, b_dense)
    call bml_export_to_dense(c, c_dense)
    call bml_export_to_dense(d, d_dense)
    call bml_print_matrix("A", a_dense, 1, n, 1, n)
    call bml_print_matrix("B", b_dense, 1, n, 1, n)
    call bml_print_matrix("C", c_dense, 1, n, 1, n)
    print *, "alpha = ", alpha
    print *, "beta = ", beta
    call bml_print_matrix("alpha * A * B + beta * C", d_dense, 1, n, 1, n)
    allocate(e_dense(n, n))
    e_dense = alpha * matmul(a_dense, b_dense) + beta * c_dense
    if(maxval(abs(e_dense - d_dense)) > abs_tol) then
      test_result = .false.
      print *, "incorrect matrix product"
      print *, "max abs diff = ", maxval(abs(e_dense - d_dense))
    endif

    call bml_export_to_dense(f, f_dense)
    call bml_export_to_dense(g, g_dense)
    call bml_print_matrix("F", f_dense, 1, n, 1, n)
    call bml_print_matrix("G", g_dense, 1, n, 1, n)
    if(maxval(abs(f_dense - g_dense)) > abs_tol) then
      test_result = .false.
      print *, "incorrect matrix product"
      print *, "max abs diff = ", maxval(abs(f_dense - g_dense))
    endif

    call bml_deallocate(a)
    call bml_deallocate(b)
    call bml_deallocate(c)
    call bml_deallocate(d)
    call bml_deallocate(f)
    call bml_deallocate(g)

  end function test_multiply_matrix_typed

end module multiply_matrix_typed
