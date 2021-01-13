NVROOT = /usr/local/cuda/
CXX = g++
CFLAGS = -Wall -Wextra -Werror -Wfatal-errors -g -m64 -O3 -MMD -MP
CXXFLAGS = $(CFLAGS) -std=c++14 -march=native -mtune=native
NVCC = $(NVROOT)/bin/nvcc
NVCCFLAGS = -Xcompiler "$(CFLAGS)" -std=c++11
LDFLAGS  += $(CXXFLAGS) -L$(NVROOT)/lib64 -lcuda -lcudart
LIBS     =
TARGET   = matmul_fp16
INCLUDE = -I/usr/local/cuda/include
OBJDIR   = ./obj
CXXSOURCES  = $(notdir $(wildcard *.cpp))
CXXOBJECTS  = $(addprefix $(OBJDIR)/, $(CXXSOURCES:.cpp=.o))
NVSOURCES  = $(notdir $(wildcard *.cu))
NVOBJECTS  = $(addprefix $(OBJDIR)/, $(NVSOURCES:.cu=.o))
DEPENDS  = $(OBJECTS:.o=.d)

# TensorCoreを使用する場合は1, 使用しない場合は1以外を指定
USE_TENSORCORE = 1
ifeq ($(USE_TENSORCORE), 1)
	CFLAGS += -DUSE_TENSORCORE
	NVCCFLAGS += -arch=sm_70
else
	NVCCFLAGS += -arch=sm_61
endif

all: $(TARGET)

clean:
	rm -rf $(OBJDIR) $(TARGET) 

#####################
$(TARGET): $(CXXOBJECTS) $(NVOBJECTS) $(LIBS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJDIR)/%.o: %.cu
	@[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ -c $< 

-include $(DEPENDS)
