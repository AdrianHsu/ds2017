all: fim.out

fim.out: error.h ResizableArray.h main.cu
	nvcc main.cu -g -o fim.out --std=c++11
