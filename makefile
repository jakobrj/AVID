
CC=g++
NVCC=nvcc

INCDIRS=-I/usr/local/cuda-11.5/include -I. -Icommon/inc -I./common/inc
LINKDIRS=-lglut -lGL -lGLEW 

CPP=

CU=src/algorithms/GPU_PROCLUS.cu src/utils/gpu_util.cu src/utils/mem_util.cpp

CUFLAGS=-arch=sm_75 --extended-lambda --ptxas-options=-v

release: DV.cu
	$(NVCC) -o bin/release/main DV.cu $(CPP) $(CU) $(INCDIRS) $(LINKDIRS) $(CUFLAGS) -O3

run:
	./bin/release/main


build_and_run: DV.cu
	$(NVCC) -o bin/release/main DV.cu $(CPP) $(CU) $(INCDIRS) $(LINKDIRS) $(CUFLAGS) -O3
	./bin/release/main