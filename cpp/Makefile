CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic
LDFLAGS = -fopenmp -lm

all: madhava_sec_benchmark

madhava_sec_benchmark: madhava_sec_benchmark.cpp madhava_core.h
	$(CXX) $(CXXFLAGS) madhava_sec_benchmark.cpp -o madhava_sec_benchmark $(LDFLAGS)

clean:
	rm -f madhava_sec_benchmark

.PHONY: all clean
