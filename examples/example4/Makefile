
BML_PATH = $(HOME)/bml/install/lib/
BML_PATH = /home/toks/CoPa/bml/install/lib64
CC = icc
CFLAGS = -I${BML_PATH}/../include -fopenmp 
LINK = -L${BML_PATH} -lbml -lbml_fortran  -llapack -lblas -lm

#For the Cray environment:
#BML_PATH = $(SCRATCH)/bml/install/lib64
#CC = cc
#CFLAGS = -I${BML_PATH}/../include -qopenmp -dynamic
#LINK = -g -dynamic -qopt-report=5 -axMIC-AVX512,CORE-AVX2 -L${BML_PATH} -lbml -lbml_fortran  -lm

main: main.o 
	     $(CC) $(CFLAGS) -o main main.o $(LINK)
