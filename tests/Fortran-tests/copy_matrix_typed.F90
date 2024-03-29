module copy_matrix_typed

  use bml
  use prec
  use bml_copy_m

  implicit none

  public :: test_copy_matrix_typed

contains

  function test_copy_matrix_typed(matrix_type, element_kind, &
       & element_precision, n, m) result(test_result)

    character(len=*), intent(in) :: matrix_type, element_kind
    integer, intent(in) :: element_precision
    integer, intent(in) :: n, m
    logical :: test_result
    character(20) :: bml_dmode

    type(bml_matrix_t) :: a
    type(bml_matrix_t) :: b
    type(bml_matrix_t) :: c

    DUMMY_KIND(DUMMY_PREC), allocatable :: a_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: b_dense(:, :)
    DUMMY_KIND(DUMMY_PREC), allocatable :: c_dense(:, :)

    if (bml_getNRanks().gt.1)then
      print*,'BML_DMODE_DISTRIBUTED'
      bml_dmode = BML_DMODE_DISTRIBUTED
    else
      print*,'BML_DMODE_SEQUENTIAL'
      bml_dmode = BML_DMODE_SEQUENTIAL
    endif

    call bml_random_matrix(matrix_type, element_kind, element_precision, n, m, &
         & a, bml_dmode)
    call bml_copy_new(a, b)
    call bml_copy_new(b, c)

    call bml_export_to_dense(a, a_dense)
    call bml_export_to_dense(b, b_dense)
    call bml_export_to_dense(c, c_dense)
    if (bml_getMyRank().eq.0)then
      call bml_print_matrix("A", a_dense, 1, n, 1, n)
      call bml_print_matrix("B", b_dense, 1, n, 1, n)
      call bml_print_matrix("C", c_dense, 1, n, 1, n)
      if(maxval(abs(a_dense - b_dense)) > 1e-12 .or. &
           maxval(abs(a_dense - c_dense)) > 1e-12) then
        test_result = .false.
        print *, "matrices are not identical"
      else
        test_result = .true.
      end if
    else
      test_result = .true.
    endif

    call bml_deallocate(a)
    call bml_deallocate(b)
    call bml_deallocate(c)

    if (bml_getMyRank().eq.0)then
      deallocate(a_dense)
      deallocate(b_dense)
      deallocate(c_dense)
    endif

  end function test_copy_matrix_typed

end module copy_matrix_typed
