  !> Matrix diagonalization functions.
module bml_diagonalize_dense_m
  implicit none

#ifdef HAVE_DSYEV
  interface dsyev
     subroutine dsyev(JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO)
       character :: JOBZ
       character :: UPLO
       integer :: N
       double precision, dimension(LDA, *) :: A
       integer :: LDA
       double precision, dimension(*) :: W
       double precision, dimension(*) :: WORK
       integer :: LWORK
       integer :: INFO
     end subroutine dsyev
  end interface dsyev
#endif

contains

  !> Diagonalize a matrix.
  !!
  !! @param a The matrix.
  !! @param eigenvectors The set of eigenvectors.
  !! @param eigenvalues The corresponding eigenvalues.
  subroutine diagonalize_dense(a, eigenvectors, eigenvalues)

    use bml_type_m
    use bml_type_dense_m
    use bml_error_m

    class(bml_matrix_dense_t), intent(in) :: a
    double precision, allocatable, intent(out) :: eigenvectors(:, :)
    double precision, allocatable, intent(out) :: eigenvalues(:)

    double precision, allocatable :: a_temp(:, :)
    double precision, allocatable :: work(:)
    integer :: lwork
    integer :: info

    select type(a)
    class is(bml_matrix_dense_double_t)
#ifdef HAVE_DSYEV
       eigenvectors = a%matrix
       allocate(eigenvalues(a%n))
       lwork = max(1, 3*a%n-1)
       allocate(work(lwork))
       call dsyev("V", "U", a%n, eigenvectors, a%n, eigenvalues, work, lwork, info)
       if(info /= 0) then
          call bml_error(__FILE__, __LINE__, "dsyev returned an error")
       end if
       deallocate(work)
#else
       call bml_error(__FILE__, __LINE__, "could not find LAPACK(dsyev) during configuration")
#endif
    class default
       call bml_error(__FILE__, __LINE__, "unknow matrix type")
    end select

  end subroutine diagonalize_dense

end module bml_diagonalize_dense_m
