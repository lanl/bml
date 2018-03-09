#Assumes environment variable PKG_CONFIG_PATH includes
#path to bml.pc, for instance, with a cshell
#setenv PKG_CONFIG_PATH $PKG_CONFIG_PATH":$HOME/GIT/bml/install/lib64/pkgconfig"

CXXFLAGS += `pkg-config --cflags bml`
LIBS+=`pkg-config --libs bml` -fopenmp

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
	
