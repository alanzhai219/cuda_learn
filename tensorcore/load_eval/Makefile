NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_70 -Xptxas -dlcm=cg
TARGET=load_eval

$(TARGET):main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
