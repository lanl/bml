module multiply_matrix_m

  use bml
  use test_m

  implicit none

  private

  type, public, extends(test_t) :: multiply_matrix_t
   contains
     procedure, nopass :: test_function
  end type multiply_matrix_t

contains

  function test_function(matrix_type, matrix_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type
    character(len=*), intent(in) :: matrix_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b
    type(bml_matrix_t) :: c
    type(bml_matrix_t) :: d

    REAL_TYPE, allocatable :: a_dense(:, :)
    REAL_TYPE, allocatable :: b_dense(:, :)
    REAL_TYPE, allocatable :: c_dense(:, :)
    REAL_TYPE, allocatable :: d_dense(:, :)
    REAL_TYPE, allocatable :: e_dense(:, :)

    double precision :: alpha = -0.8
    double precision :: beta = 1.2

#if defined(SINGLE_REAL) || defined(SINGLE_COMPLEX)
    double precision :: abs_tol = 1e-6
#else
    double precision :: abs_tol = 1d-12
#endif
    call bml_random_matrix(matrix_type, matrix_precision, n, m, a)
    call bml_identity_matrix(matrix_type, matrix_precision, n, m, b)
    call bml_identity_matrix(matrix_type, matrix_precision, n, m, c)
    call bml_copy(c, d)

    call bml_multiply(a, b, d, alpha, beta)

    test_result = .true.

    call bml_convert_to_dense(a, a_dense)
    call bml_convert_to_dense(b, b_dense)
    call bml_convert_to_dense(c, c_dense)
    call bml_convert_to_dense(d, d_dense)
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

    call bml_deallocate(a)
    call bml_deallocate(b)
    call bml_deallocate(c)
    call bml_deallocate(d)

  end function test_function

end module multiply_matrix_m
