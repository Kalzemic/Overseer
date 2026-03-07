CXX  := g++

CXXFLAGS := -std=c++17 -Wall -Wextra -O2
LDFLAGS  := -lbpf -lelf -lz -pthread


SRC      := main.cpp Transport.cpp
OBJ      := $(SRC:.cpp=.o)
TARGET   := overseer_collector


all: skeleton $(TARGET)

skeleton: 
	clang -O2 -target bpf -g -c FaultCollect.bpf.c -o FaultCollect.bpf.o 
	bpftool gen skeleton FaultCollect.bpf.o > skel.bpf.h




$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean: 
	rm -f FaultCollect.bpf.o $(OBJ) $(TARGET)