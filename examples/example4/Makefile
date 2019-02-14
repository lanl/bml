MAGMA_PATH = $(HOME)/magma-2.5.0/
BML_PATH = $(HOME)/bml/install/lib64/
CC = xlc_r
CFLAGS = -I${BML_PATH}/../include -I${MAGMA_PATH}/include -I${MAGMA_PATH}/lib -qsmp=omp 
LINK = -L${BML_PATH} -lbml -lbml_fortran -L${OLCF_ESSL_ROOT}/lib64 -lesslsmp -lm
LINK += -L${MAGMA_PATH}/lib/ -lmagma 


main: main.o 
	     $(CC) $(CFLAGS) -o main main.o $(LINK)
