# Builds the VMC depinning-field solver (main.cu + misistema.h).
#
# Model parameters are compile-time constants (paper sets c=eps0=1
# without loss of generality); override on the command line, e.g.:
#   make AMPDIS=0.4 CEL=1.0
# TARGET names the binary (to keep one per Delta) and ARCH the GPU
# architecture, e.g.:
#   make AMPDIS=0.6 TARGET=phi4vmc_D0.6 ARCH=sm_86
AMPDIS=0.2
CEL=1.0
EPSILON0=1.0
R0=1.0

FLAGS=-DAMPDIS=$(AMPDIS) -DCEL=$(CEL) -DEPSILON0=$(EPSILON0) -DR0=$(R0)

TARGET=phi4vmc
ARCH=sm_61

NVCC=-std=c++14 -arch=$(ARCH)

$(TARGET): main.cu misistema.h
	nvcc $(NVCC) -O2 -I. -o $(TARGET) main.cu $(FLAGS)

clean:
	rm -f phi4vmc phi4vmc_D*
