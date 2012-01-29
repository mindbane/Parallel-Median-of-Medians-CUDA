run: main.cu  main.h  median_kernel.cu  median_kernel.h 
	nvcc -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20 --compiler-options -fno-strict-aliasing -I/opt/gpu_sdk/C/common/inc -L/opt/gpu_sdk/C/lib -lcutil_x86_64 -DUNIX median_kernel.cu main.cu -o run -O3
