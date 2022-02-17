
CC=g++
NVCC=nvcc

INCDIRS=-I/usr/local/cuda-11.5/include -I. -Icommon/inc 
LINKDIRS=-lglut -lGL -lGLEW

CPP=

CU=src/algorithms/GPU_PROCLUS.cu src/utils/gpu_util.cu

CUFLAGS=-arch=sm_75 --extended-lambda --ptxas-options=-v

release: src/DV.cu
	$(NVCC) -o bin/release/main src/DV.cu $(CPP) $(CU) $(INCDIRS) $(LINKDIRS) $(CUFLAGS) -O3

run_release:
	python generate.py $(n) $(d) $(cl)
	./bin/release/main $(n) $(d) $(cl) $(v)
