CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic -fPIC
LDFLAGS = -fopenmp -lm

all: madhava_sec_benchmark libmadhava_sec.so

madhava_sec_benchmark: madhava_sec_benchmark.cpp madhava_core.h
	$(CXX) $(CXXFLAGS) madhava_sec_benchmark.cpp -o madhava_sec_benchmark $(LDFLAGS)

# Shared library (C ABI) for the native engine — loaded by Python via ctypes.
libmadhava_sec.so: madhava_sec_capi.cpp madhava_core.h
	$(CXX) $(CXXFLAGS) -shared madhava_sec_capi.cpp -o libmadhava_sec.so $(LDFLAGS)

clean:
	rm -f madhava_sec_benchmark libmadhava_sec.so

.PHONY: all clean
