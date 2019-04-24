all:
	g++ HW5_opencl.cpp -o HW5_opencl -lOpenCL
	nvcc Jacobi_CUDA.cu -o Jacobi_CUDA
	gcc -std=c99 Jacobi_Serial.c -lm -o Jacobi_Serial
