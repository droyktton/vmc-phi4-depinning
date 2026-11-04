# Builds the VMC depinning-field solver (main.cu + misistema.h).
#
# Model parameters are compile-time constants (paper sets c=eps0=1
# without loss of generality); override on the command line, e.g.:
#   make AMPDIS=0.4 CEL=1.0
# TARGET names the binary (to keep one per Delta), e.g.:
#   make AMPDIS=0.6 TARGET=phi4vmc_D0.6
# anisotropic (block-correlated) disorder, e.g.:
#   make AMPDIS=0.3 AX=4 AY=1 TARGET=phi4vmc_D0.3_ax4ay1
#
# The GPU architecture is picked automatically: on a machine with a
# visible GPU (e.g. a laptop), its compute capability (nvidia-smi); with
# no GPU (e.g. a cluster frontend), a multi-arch binary for CLUSTER_ARCHS.
# Force one with ARCH, e.g. `make ARCH=sm_86`.
AMPDIS=0.2
CEL=1.0
EPSILON0=1.0
R0=1.0
# disorder correlated over AX x AY blocks of sites (1x1: uncorrelated)
AX=1
AY=1

FLAGS=-DAMPDIS=$(AMPDIS) -DCEL=$(CEL) -DEPSILON0=$(EPSILON0) -DR0=$(R0) -DAX=$(AX) -DAY=$(AY)

TARGET=phi4vmc

# GPUs of the cluster's gpu partition: RTX 3080/3080 Ti/A10 (sm_86), RTX 5070 Ti (sm_120)
CLUSTER_ARCHS=86 120

# Thrust from CUDA >= 13 requires C++17; CUDA < 11 doesn't support it
NVCC_MAJOR:=$(shell nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
STD:=$(if $(filter 9 10,$(NVCC_MAJOR)),c++14,c++17)

LOCAL_CC:=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.[:space:]' | grep -E '^[0-9]+$$')
ifdef ARCH
GENCODE=-arch=$(ARCH)
else ifneq ($(LOCAL_CC),)
GENCODE=-arch=sm_$(LOCAL_CC)
else
GENCODE=$(foreach a,$(CLUSTER_ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
endif

NVCC=-std=$(STD) $(GENCODE)

$(TARGET): main.cu misistema.h
	nvcc $(NVCC) -O2 -I. -o $(TARGET) main.cu $(FLAGS)

# velocity-driven ensemble (main_vd.cu), e.g. make vd AMPDIS=0.3 VD_TARGET=phi4vmc_vd_D0.3
VD_TARGET=phi4vmc_vd

vd: $(VD_TARGET)

$(VD_TARGET): main_vd.cu misistema.h
	nvcc $(NVCC) -O2 -I. -o $(VD_TARGET) main_vd.cu $(FLAGS)

# the same at finite velocity, Euler dynamics (main_vdyn.cu), e.g. make vdyn AMPDIS=0.3 VDYN_TARGET=phi4vmc_vdyn_D0.3
VDYN_TARGET=phi4vmc_vdyn

vdyn: $(VDYN_TARGET)

$(VDYN_TARGET): main_vdyn.cu misistema.h
	nvcc $(NVCC) -O2 -I. -o $(VDYN_TARGET) main_vdyn.cu $(FLAGS)

# SYCL port for Intel GPUs (misistema_sycl.h; oneAPI icpx + oneDPL), e.g.
#   module swap gnu13 intel/2025.3.0; module load dpl
#   make sycl AMPDIS=0.4 TARGET_SYCL=phi4vmc_sycl_D0.4
# Compiled ahead of time for SYCL_AOT (pvc: Data Center GPU Max); with
# SYCL_AOT= (empty) it is JIT-compiled instead, so it runs on any SYCL
# device (e.g. the CPU, with ONEAPI_DEVICE_SELECTOR=opencl:cpu).
# -fp-model=precise: icpx defaults to fast math, which would change the
# roots of the cubic and the convergence tests.
TARGET_SYCL=phi4vmc_sycl
SYCL_AOT=pvc
SYCL_TARGETS:=$(if $(SYCL_AOT),-fsycl-targets=spir64_gen -Xsycl-target-backend "-device $(SYCL_AOT)")

sycl: $(TARGET_SYCL)

$(TARGET_SYCL): main.cu misistema_sycl.h
	icpx -fsycl $(SYCL_TARGETS) -std=c++17 -O2 -fp-model=precise -I. -DVMC_SYCL -x c++ -o $(TARGET_SYCL) main.cu $(FLAGS)

clean:
	rm -f phi4vmc phi4vmc_D* phi4vmc_vd* phi4vmc_sycl*

.PHONY: vd vdyn sycl clean
