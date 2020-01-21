module normalize_matrix_typed

  use bml
  use prec
  use bml_normalize_m

  implicit none

  public :: test_normalize_matrix_typed

contains

  function test_normalize_matrix_typed(matrix_type, element_kind, element_precision, n, m) &
       & result(test_result)

    character(len=*), intent(in) :: matrix_type, element_kind
    integer, intent(in) :: element_precision
    integer, intent(in) :: n, m
    logical :: test_result

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b

    real(dp), allocatable :: a_gbnd(:)
    real(dp), allocatable :: b_gbnd(:)
    real(dp) :: scale_factor, threshold
    real(DUMMY_PREC) :: rel_tol

    DUMMY_KIND(DUMMY_PREC), allocatable :: a_dense(:,:)
    DUMMY_KIND(DUMMY_PREC), allocatable :: b_dense(:,:)

    integer :: i, j

    if(element_precision == sp)then
      rel_tol = 1e-6
    elseif(element_precision == dp)then
      rel_tol = 1d-12
    endif

    test_result = .true.

    scale_factor = 2.5_dp
    threshold = 0.0_dp

    call bml_identity_matrix(matrix_type, element_kind, element_precision, n, &
         & m, a)
    call bml_scale(scale_factor, a)
    call bml_gershgorin(a, a_gbnd)
    call bml_export_to_dense(a, a_dense)
    a_dense(1,1) = scale_factor * scale_factor
    call bml_import_from_dense(matrix_type, a_dense, b, threshold, m)
    call bml_gershgorin(b, b_gbnd);
    write(*,*) 'B maxeval = ', b_gbnd(1), ' maxminusmin = ', b_gbnd(2)

    call bml_export_to_dense(a, a_dense);
    call bml_export_to_dense(b, b_dense);

    call bml_print_matrix("A", a_dense, lbound(a_dense, 1), ubound(a_dense, 1), &
         lbound(a_dense, 2), ubound(a_dense, 2))
    call bml_print_matrix("B", b_dense, lbound(b_dense, 1), ubound(b_dense, 1), &
         lbound(b_dense, 2), ubound(b_dense, 2))

    call bml_normalize(b, b_gbnd(1), b_gbnd(2))

    call bml_export_to_dense(b, b_dense);

    call bml_print_matrix("B", b_dense, lbound(b_dense, 1), ubound(b_dense, 1), &
         lbound(b_dense, 2), ubound(b_dense, 2))

    if ((abs(a_gbnd(1) - scale_factor) > rel_tol) .or. (a_gbnd(2) > rel_tol)) then
      print *, "Incorrect maxeval or maxminusmin"
      test_result = .false.
    end if

    if ((abs(b_gbnd(1) - scale_factor*scale_factor) > rel_tol) .or. &
         (abs(b_gbnd(2) - (scale_factor*scale_factor - scale_factor)) > rel_tol)) then
      print *, "Incorrect maxeval or maxminusmin"
      test_result = .false.
    end if

    if (abs(b_dense(1,1)) > rel_tol) then
      print *, "Incorrect maxeval or maxminusmin, failed normalize"
      test_result = .false.
    end if

    if(test_result) then
      print *, "Test passed"
    end if

    call bml_deallocate(a)
    call bml_deallocate(b)

    deallocate(a_dense)
    deallocate(b_dense)
    deallocate(a_gbnd)
    deallocate(b_gbnd)

  end function test_normalize_matrix_typed

end module normalize_matrix_typed
