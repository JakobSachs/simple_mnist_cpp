CC=gcc
CXX=g++
RM=rm -f
# include openmp and -Xpreprocessor -fopenmp are needed for cpp on Mac chips
EIGEN_CFLAGS=$(shell pkg-config eigen3 --cflags)
CPPFLAGS=-g -Xpreprocessor -fopenmp $(EIGEN_CFLAGS) -std=c++2b -Og -I/opt/homebrew/opt/libomp/include
# -L/opt/homebrew/opt/libomp/lib -lomp is needed for cpp on Mac chips
LDFLAGS=-g -L/opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib -lomp

SRCS=main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: main

main: $(OBJS)
	$(CXX) $(LDFLAGS) -o main $(OBJS) 

main.o: main.cpp
	$(CXX) $(CPPFLAGS) -c main.cpp

clean:
	$(RM) main $(OBJS)


