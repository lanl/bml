module diagonalize_matrix_typed

  use bml
  use prec
  use bml_diagonalize_m

  implicit none

  public :: test_diagonalize_matrix_typed

contains

  function test_diagonalize_matrix_typed(matrix_type, element_kind, element_precision, n, m) &
       & result(test_result)

    character(len=*), intent(in) :: matrix_type, element_kind
    integer, intent(in) :: element_precision
    integer, intent(in) :: n, m
    logical :: test_result
    real(dp) :: threshold

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: a_t
    type(bml_matrix_t) :: b
    type(bml_matrix_t) :: c
    type(bml_matrix_t) :: eigenvectors
    type(bml_matrix_t) :: eigenvectors_t
    !  DUMMY_KIND(DUMMY_PREC), allocatable :: eigenvalues(:)
    real(8), allocatable :: eigenvalues(:)

    write(*,*)"element ",element_kind,element_precision
    threshold = 0.0_MP
    test_result = .false.

    call bml_random_matrix(matrix_type, element_kind, element_precision, n, m, &
         & a)
    call bml_print_matrix("A", a, 1, n, 1, n)
    call bml_transpose(a, a_t)
    call bml_print_matrix("A_t", a_t, 1, n, 1, n)
    call bml_add(a, a_t, 0.5d0, 0.5d0, threshold)
    call bml_print_matrix("A", a, 1, n, 1, n)
    allocate(eigenvalues(n))
    call bml_zero_matrix(matrix_type, element_kind, element_precision, n, m, &
         & eigenvectors)
    !! \todo Fixme: diagonalization routine is not respecting precision
    call bml_diagonalize(a, eigenvalues, eigenvectors)
    call bml_transpose(eigenvectors, eigenvectors_t)
    call bml_zero_matrix(matrix_type, element_kind, element_precision, n, m, b)
    call bml_zero_matrix(matrix_type, element_kind, element_precision, n, m, c)
    call bml_multiply(eigenvectors_t, eigenvectors, b)
    call bml_multiply(b, eigenvectors, c)
    write(*, *) eigenvalues
    call bml_print_matrix("eigenvectors", eigenvectors, 1, n, 1, n)
    call bml_print_matrix("U^t U", b, 1, n, 1, n)
    test_result = .false.

    call bml_deallocate(a)
    call bml_deallocate(a_t)
    call bml_deallocate(b)
    call bml_deallocate(c)
    call bml_deallocate(eigenvectors)
    call bml_deallocate(eigenvectors_t)

    write(*,*)test_result

    deallocate(eigenvalues)

  end function test_diagonalize_matrix_typed

end module diagonalize_matrix_typed
