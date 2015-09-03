!> \copyright Los Alamos National Laboratory 2015

!> Matrix initialization.
module bml_allocate_m
  implicit none
contains

  !> Allocate a matrix.
  !!
  !! \ingroup allocate_group
  !!
  !! \param matrix_type The matrix type.
  !! \param N The matrix size.
  !! \param A The matrix.
  !! \param matrix_precision The precision of the matrix. The default
  !! is double precision.
  subroutine bml_allocate(matrix_type, N, A, matrix_precision)

    use bml_type_m
    use bml_allocate_dense_m
    use bml_allocate_ellpack_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(bml_matrix_t), allocatable, intent(out) :: A
    character(len=*), optional, intent(in) :: matrix_precision

    character(len=:), allocatable :: matrix_precision_

    if(present(matrix_precision)) then
       matrix_precision_ = matrix_precision
    else
       matrix_precision_ = BML_PRECISION_DOUBLE
    end if

    select case(matrix_type)
    case(BML_MATRIX_DENSE)
       call allocate_matrix_dense(N, A, matrix_precision_)
    case(BML_MATRIX_ELLPACK)
       call allocate_matrix_ellpack(N, A, matrix_precision_)
    case default
       call bml_error(__FILE__, __LINE__, "unsupported matrix type ("//trim(matrix_type)//")")
    end select

  end subroutine bml_allocate

  !> Deallocate a matrix.
  !!
  !! \ingroup allocate_group
  !!
  !! \bug This procedure should be called even if the matrix object is
  !! implicitly de-allocated, i.e. when it goes of out scope. This
  !! behavior might depend on the complier, since it's a fairly recent
  !! Fortran standard addition, and not all compilers implement such a
  !! thing currently.
  !!
  !! \param A The matrix.
  subroutine bml_deallocate(A)

    use bml_type_m
    use bml_type_dense_m
    use bml_type_ellpack_m
    use bml_allocate_dense_m
    use bml_allocate_ellpack_m
    use bml_error_m

    class(bml_matrix_t), allocatable, intent(inout) :: A

    if(allocated(A)) then
       select type(A)
       class is(bml_matrix_dense_t)
          call deallocate_matrix_dense(A)
       class is(bml_matrix_ellpack_t)
          call deallocate_matrix_ellpack(A)
       class default
          call bml_error(__FILE__, __LINE__, "unsupported matrix type")
       end select
       deallocate(A)
    endif

  end subroutine bml_deallocate

  !> Initialize a random matrix.
  !!
  !! \ingroup allocate_group
  !!
  !! \param matrix_type The matrix type.
  !! \param N The matrix size.
  !! \param A The matrix.
  !! \param matrix_precision The precision of the matrix. The default
  !! is double precision.
  subroutine bml_random_matrix(matrix_type, N, A, matrix_precision)

    use bml_type_m
    use bml_allocate_dense_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(bml_matrix_t), allocatable, intent(out) :: A
    character(len=*), optional, intent(in) :: matrix_precision

    character(len=:), allocatable :: matrix_precision_

    if(present(matrix_precision)) then
       matrix_precision_ = matrix_precision
    else
       matrix_precision_ = BML_PRECISION_DOUBLE
    end if

    select case(matrix_type)
    case(BML_MATRIX_DENSE)
       call random_matrix_dense(n, a, matrix_precision_)
    case default
       call bml_error(__FILE__, __LINE__, "unsupported matrix type ("//trim(matrix_type)//")")
    end select

  end subroutine bml_random_matrix

  !> Initialize a identity matrix.
  !!
  !! \ingroup allocate_group
  !!
  !! \param matrix_type The matrix type.
  !! \param N The matrix size.
  !! \param A The matrix.
  !! \param matrix_precision The precision of the matrix. The default
  !! is double precision.
  subroutine bml_identity_matrix(matrix_type, N, A, matrix_precision)

    use bml_type_m
    use bml_allocate_dense_m
    use bml_allocate_ellpack_m
    use bml_error_m

    character(len=*), intent(in) :: matrix_type
    integer, intent(in) :: N
    class(bml_matrix_t), allocatable, intent(out) :: A
    character(len=*), optional, intent(in) :: matrix_precision

    character(len=:), allocatable :: matrix_precision_

    if(present(matrix_precision)) then
       matrix_precision_ = matrix_precision
    else
       matrix_precision_ = BML_PRECISION_DOUBLE
    end if

    select case(matrix_type)
    case(BML_MATRIX_DENSE)
       call identity_matrix_dense(n, a, matrix_precision_)
    case(BML_MATRIX_ELLPACK)
       call identity_matrix_ellpack(n, a, matrix_precision_)
    case default
       call bml_error(__FILE__, __LINE__, "unsupported matrix type ("//trim(matrix_type)//")")
    end select

  end subroutine bml_identity_matrix

end module bml_allocate_m
