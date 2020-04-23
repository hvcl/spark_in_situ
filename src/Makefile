all:
	#Only for CUDA 
	nvcc -o ../bin/gpu_manager c_manager.cu -std=c++11 -O3 -lcuda -Xcompiler -fopenmp -lmpi -L/usr/lib/openmpi/lib -I/usr/lib/openmpi/include 
	#With OpenGL
	#nvcc -o ../bin/gpu_manager c_manager.cu -std=c++11 -O3 -lcuda -Xcompiler -fopenmp -lmpi -lmpi_cxx -L/usr/lib/openmpi/lib -I/usr/lib/openmpi/include -lEGL -lGL -lGLU -lGLEW -D_USE_EGL_


