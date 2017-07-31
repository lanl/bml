program main

  use bml

  implicit none
  integer, parameter :: dp = kind(1.0d0)
  type(bml_matrix_t) :: hmat_bml
  real(dp), allocatable :: H(:,:)
  real(dp), allocatable :: row(:),diag(:),mydiag(:),myrow(:)
  character(20) :: bml_type
  real(dp) :: numthresh
  integer :: i,j,hdim,mdim


  write(*,*)''
  write(*,*)'Example Fortran BML 3: Read a dense matrix form file, convert to bml_dense'
  write(*,*)'; get row and get diagonal'
  write(*,*)'This can be also done for dense by changing the ellpack type'
  write(*,*)''

  open(1,file='hamiltonian.dat')
  read(1,*)hdim
  mdim = hdim/2
  bml_type = "dense"
  numthresh = 0.05_dp

  allocate(H(hdim,hdim))
  allocate(row(hdim))
  allocate(diag(hdim))
  allocate(myrow(hdim))
  allocate(mydiag(hdim))

  H=0.0_dp

  do i=1,hdim
     do j=1,hdim
        read(1,*)H(i,j)
     enddo
     myrow(i) = real(i,dp)
     mydiag(i) = real(i,dp)
  enddo

  call bml_zero_matrix(bml_type,bml_element_real,dp,hdim,mdim,hmat_bml)

  call bml_convert_from_dense(bml_type,H,hmat_bml,numthresh,mdim)

  call bml_get_diagonal(hmat_bml,diag)

  !Get the third row
  call bml_get_row(hmat_bml,3,row)

  !The printing routine starts from 0.
  call bml_print_matrix("hmat_bml",hmat_bml,0,6,0,6)

  write(*,*)"diagonal = (",diag(1),diag(2),diag(3),"...)"
  write(*,*)"third row = (",row(1),row(2),row(3),"...)"

end program main
