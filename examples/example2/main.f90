program main

  use bml

  implicit none
  integer, parameter :: dp = kind(1.0d0)
  type(bml_matrix_t) :: hmat_bml
  complex(dp), allocatable :: H(:,:)
  character(20) :: bml_type
  real(dp) :: numthresh, var
  complex(dp) :: scalar
  integer :: i,j,hdim,mdim

  write(*,*)''
  write(*,*)'Example Fortran BML 2: Read a dense matrix form file, convert to complex bml_dense'
  write(*,*)'; multiply by a complex scalar'
  write(*,*)''

  open(1,file='hamiltonian.dat')
  read(1,*)hdim
  mdim = hdim/2
  bml_type = "dense"
  numthresh = 0.05_dp

  allocate(H(hdim,hdim))

  H=0.0_dp

  do i=1,hdim
     do j=1,hdim
        read(1,*)var
        H(i,j)=var*cmplx(1.0_dp,0.0_dp)
     enddo
  enddo

  call bml_zero_matrix(bml_type,bml_element_complex,dp,hdim,mdim,hmat_bml)

  call bml_convert_from_dense(bml_type,H,hmat_bml,numthresh,mdim)

  scalar = cmplx(2.0_dp,2.0_dp)

  call bml_scale(scalar,hmat_bml)

  write(*,*)''
  write(*,*)"bml_type = ", bml_get_type(hmat_bml)
  write(*,*)"N = ", bml_get_N(hmat_bml)
  write(*,*)"M = ", bml_get_M(hmat_bml)
  write(*,*)''

  !The printing routine starts from 0.
  call bml_print_matrix("hmat_bml",hmat_bml,0,6,0,6)

end program main
