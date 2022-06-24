#Assumes environment variable PKG_CONFIG_PATH includes
#path to bml.pc, for instance, with a cshe

CXXFLAGS += -I/$(HOME)/bml/install/include
LIBS+= -L/$(HOME)/bml/install/lib64/ -fopenmp -lbml -lm
LIBS+= -L$(CUDA_LIB) -lcublas -lcuda -lcudart 
# how to produce objects from source files

%.o:%.cc
	$(CXX) $(CXXFLAGS) -c $<

TARGET = ./main

all: $(TARGET)

OBJS = main.o 

# dependencies
$(TARGET): $(OBJS)

# rules

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

clean:
	rm -f *.o
	rm -f main
	
