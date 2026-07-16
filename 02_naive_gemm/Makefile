# 若 -arch=native 报错（旧 nvcc），改成：ARCH = -arch=sm_89
NVCC ?= nvcc
ARCH ?= -arch=native
FLAGS = -O3 -std=c++17 $(ARCH) -I../common -lcublas

all: sgemm

sgemm: sgemm.cu ../common/helpers.cuh
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f sgemm

.PHONY: all clean
