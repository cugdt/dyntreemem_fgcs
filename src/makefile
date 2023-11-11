###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=nvcc
CC_FLAGS=
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS=
NVCC_LIBS=

##########################################################

## Project file structure ##

# Source file directory:
#SRC_DIR = src
SRC_DIR = .

# Object file directory:
#OBJ_DIR = bin
OBJ_DIR = .

# Include header file directory:
#INC_DIR = src
INC_DIR = .

##########################################################

## Make variables ##

# Target executable name:
EXE = mlpdt

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/AttrValue.o $(OBJ_DIR)/CudaTimeStats.o $(OBJ_DIR)/CudaWorker.o $(OBJ_DIR)/CudaWorkerRepTest.o $(OBJ_DIR)/DTreeNodeSim.o $(OBJ_DIR)/IEvolutionaryAlg.o $(OBJ_DIR)/ITest.o $(OBJ_DIR)/RealTest.o $(OBJ_DIR)/Worker.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)