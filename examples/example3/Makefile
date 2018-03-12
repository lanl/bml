BML_DIR=$(HOME)/bml/install
F90      = gfortran 
F90FLAGS += -fopenmp -I/$(BML_DIR)/include -L/$(BML_DIR)/lib/ -lbml_fortran -lbml
LIBS+= -llapack -lblas -fopenmp
LIBS+= -L/$(BML_DIR)/lib/ -lbml_fortran -lbml

# how to produce objects from source files

%.o:%.f90
	$(F90) $(F90FLAGS) -c $<

%.o:%.f95
	$(F90) $(F90FLAGS) -c $<

%.o:%.f
	$(F77) $(F77FLAGS) -c $<

# General vars

TARGET = ./main

all: $(TARGET)

PROGRAM = main.o

OBJS = $(PROGRAM) $(MODULES)

# dependencies

$(MODULES) :
$(PROGRAM) : $(MODULES)

# rules

$(TARGET):	$(MODULES) $(PROGRAM)
	$(F90) -o  $(TARGET) $(LINKFLAGS) $(OBJS) $(LIBS) $(LIBSP)

clean:
	rm -f *.o *.mod 
	rm -f main
