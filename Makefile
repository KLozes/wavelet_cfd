TARGET = compEuler

## compilers
CC = g++
NVCC = nvcc

# include directories
INC_DIR = -I./src

# source directory
SRC_DIR = src

# object directory
OBJ_DIR = obj

# c++ source files
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# CUDA source files
CU_SRCS  = $(wildcard $(SRC_DIR)/*.cu)

# objects
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.cpp.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_SRCS))

OBJS =  $(CPP_OBJS)
OBJS += $(CU_OBJS)

## compile options
CPPFLAGS = -O3 -Wextra -std=c++14
NVCCFLAGS =  -std=c++14 -arch=sm_61
NVCCLFLAGS =  -std=c++14 -arch=sm_61

## Build Rules
$(TARGET) : $(OBJS)
	$(NVCC) $(NVCCLFLAGS) $(OBJS) -o $(TARGET)

$(OBJ_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) $(INC_DIR) -c $< -o $@

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INC_DIR) -c $< -o $@

clean:
	\rm obj/*.o $(TARGET)
