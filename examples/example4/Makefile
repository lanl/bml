
BML_DIR=$(HOME)/bml/install
F90      = gfortran
F90FLAGS += -fopenmp -I/$(BML_DIR)/include -L/$(BML_DIR)/lib/ -lbml
LIBS+= -Wl,--no-as-needed  -L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_core  -lmkl_gnu_thread -ldl -lpthread -lm -fopenmp 
LIBS+= -L/$(BML_DIR)/lib/ -lbml


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
	rm -f *.o *.mod  work.pcl
	rm -f *.txt
#	rm -f *.dat
	rm -f *.ps
	rm -f *.out
	rm -f fort.* *.backup
	rm -f main
	rm -f *~
#	rm -f *.inp
	rm -f *.pdf
	
