
BML_PATH = $(HOME)/bml/install/lib/
CC = gcc
CFLAGS = -I${BML_PATH}/../include -fopenmp 
LINK = -L${BML_PATH} -lbml -lbml_fortran  -llapack -lblas -lm

main: main.o 
	     $(CC) $(CFLAGS) -o main main.o $(LINK)
