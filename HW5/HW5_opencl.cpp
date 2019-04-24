/*************************************************************************
	> File Name: HW5_opencl.cpp
	> Author: Ao Cai
	> Mail: aocai166@gmail.com 
	> Created Time: Sun 21 Apr 2019 11:56:02 AM CDT
 ************************************************************************/

#include<iostream>
using namespace std;
// for file IO
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <sys/stat.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/OpenCl.h>
#else
#include <CL/cl.h>
#endif

#define PI 3.14159265359f
#define MAX(a,b) (((a)>(b))?(a):(b))

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data){
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

void oclInit(int plat, int dev,
	     cl_context &context,
	     cl_device_id &device,
	     cl_command_queue &queue){

  /* set up CL */
  cl_int            err;
  cl_platform_id    platforms[100];
  cl_uint           platforms_n;
  cl_device_id      devices[100];
  cl_uint           devices_n ;

  /* get list of platform IDs (platform == implementation of OpenCL) */
  clGetPlatformIDs(100, platforms, &platforms_n);
  
  if( plat > platforms_n) {
    printf("ERROR: platform %d unavailable \n", plat);
    exit(-1);
  }
  
  // find all available device IDs on chosen platform (could restrict to CPU or GPU)
  cl_uint dtype = CL_DEVICE_TYPE_ALL;
  clGetDeviceIDs( platforms[plat], dtype, 100, devices, &devices_n);
  
  printf("devices_n = %d\n", devices_n);
  printf("dev = %d\n", dev);
  
  if(dev>=devices_n){
    printf("invalid device number for this platform\n");
    exit(0);
  }

  // choose user specified device
  device = devices[dev];
  
  // make compute context on device, pass in function pointer for error messaging
  context = clCreateContext((cl_context_properties *)NULL, 1, &device, &pfn_notify, (void*)NULL, &err); 

  // create command queue
  queue   = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); // synchronized execution
}

void oclBuildKernel(const char *sourceFileName,
		    const char *functionName,
		    cl_context &context,
		    cl_device_id &device,
		    cl_kernel &kernel,
		    const char *flags
		    ){

  cl_int            err;

  // read in text from source file
  FILE *fh = fopen(sourceFileName, "r"); // file handle
  if (fh == 0){
    printf("Failed to open: %s\n", sourceFileName);
    throw 1;
  }

  // C function, get stats for source file (just need total size = statbuf.st_size)
  struct stat statbuf; 
  stat(sourceFileName, &statbuf); 

  // read text from source file and add terminator
  char *source = (char *) malloc(statbuf.st_size + 1); // +1 for "\0" at end
  fread(source, statbuf.st_size, 1, fh); // read in 1 string element of size "st_size" from "fh" into "source"
  source[statbuf.st_size] = '\0'; // terminates the string

  // create program from source 
  cl_program program = clCreateProgramWithSource(context,
						 1, // compile 1 kernel
						 (const char **) & source,
						 (size_t*) NULL, // lengths = number of characters in each string. NULL = \0 terminated.
						 &err); 

  if (!program){
    printf("Error: Failed to create compute program!\n");
    throw 1;
  }
    
  // compile and build program 
  err = clBuildProgram(program, 1, &device, flags, (void (*)(cl_program, void*))  NULL, NULL);

  // check for compilation errors 
  char *build_log;
  size_t ret_val_size;
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size); // get size of build log
  
  build_log = (char*) malloc(ret_val_size+1);
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, (size_t*) NULL); // read build log
  
  // to be careful, terminate the build log string with \0
  // there's no information in the reference whether the string is 0 terminated or not 
  build_log[ret_val_size] = '\0';

  // print out compilation log 
  fprintf(stderr, "%s", build_log );

  // create runnable kernel
  kernel = clCreateKernel(program, functionName, &err);
  if (! kernel || err != CL_SUCCESS){
    printf("Error: Failed to create compute kernel!\n");
    throw 1;
  }
}




int main(int argc, char **argv){

	int N = atoi(argv[1]); // matrix size 
	float tol = atof(argv[2]); // tolerance

  cl_int            err;

  int plat = 1;
  int dev  = 0;

  cl_context context;
  cl_device_id device;
  cl_command_queue queue;

  cl_kernel kernel1;
  cl_kernel kernel2;

  oclInit(plat, dev, context, device, queue);

  const char *sourceFileName1 = "Jacobi.cl";
  const char *functionName1 = "Jacobi";

  const char *sourceFileName2 = "reduce.cl";
  const char *functionName2 = "reduce";

  int BDIM = 512;
  char flags[BUFSIZ];
  sprintf(flags, "-DBDIM=%d", BDIM);

  oclBuildKernel(sourceFileName1, functionName1,
		 context, device,
		 kernel1, flags);

  oclBuildKernel(sourceFileName2, functionName2,
		 context, device,
		 kernel2, flags);

  // START OF PROBLEM IMPLEMENTATION 
//	int N = 16;
//	float tol = 1e-6;

  /* create host array */
  int Nr = (N+2)*(N+2);
  size_t sz = (N+2)*(N+2)*sizeof(float);

  float *u = (float*) malloc(sz);
  float *unew = (float*) malloc(sz);
  float *b = (float*) malloc(sz);
  float *res = (float*) malloc(sz); 
  float h = 2.0/(N+1);
  clock_t begin=0, stop;
  if(plat==0){
	  begin = clock();
  }

  for (int ix = 0; ix < N+2; ix++){
	  for (int iz = 0; iz < N+2; iz++){
		  const float tmpx = -1.0 + ix*h;
		  const float tmpz = -1.0 + iz*h;
		  b[ix + iz*(N+2)] = sin(PI*tmpx)*sin(PI*tmpz)*h*h;
		  u[ix + iz*(N+2)] = 0.f;
		  unew[ix + iz*(N+2)] = 0.f;
		  res[ix + iz*(N+2)] = 0.f;		  
	  }
  }

  // create device buffer and copy from host buffer
  cl_mem d_u = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, u, &err);
  cl_mem d_unew = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, unew, &err);
  cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, b, &err);
  cl_mem d_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, res, &err);

  // now set kernel arguments
  clSetKernelArg(kernel1, 0, sizeof(int), &N);
  clSetKernelArg(kernel1, 1, sizeof(cl_mem), &d_u);
  clSetKernelArg(kernel1, 2, sizeof(cl_mem), &d_b);
  clSetKernelArg(kernel1, 3, sizeof(cl_mem), &d_unew);  
   
  clSetKernelArg(kernel2, 0, sizeof(int), &Nr);
  clSetKernelArg(kernel2, 1, sizeof(cl_mem), &d_u);
  clSetKernelArg(kernel2, 2, sizeof(cl_mem), &d_unew);
  clSetKernelArg(kernel2, 3, sizeof(cl_mem), &d_res);
  
  // set thread array 
  int dim = 1;
  int Nt = N+2;
  int Ng = N+2;
  size_t local_dims[3] = {Nt,1,1};
  size_t global_dims[3] = {Ng*Nt,1,1};

  // cl event for timing
  cl_event event;
  cl_ulong start,end;
  double nanoSeconds1 = 0.0;
  double nanoSeconds2 = 0.0;
 
  int iter = 0;
  float obj = 1.0;
  while(obj > tol*tol){

	// queue up kernel 
	clEnqueueNDRangeKernel(queue, kernel1, dim, 0, 
				 global_dims, local_dims,
				 0, (cl_event*)NULL, // wait list events
				 &event); // queue event along with kernel	

	err = clWaitForEvents(1, &event);
	clFinish(queue);

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	nanoSeconds1 += end - start;
	
    clEnqueueNDRangeKernel(queue, kernel2, dim, 0, 
				 global_dims, local_dims,
				 0, (cl_event*)NULL, // wait list events
				 &event); // queue event along with kernel
	
	err = clWaitForEvents(1, &event);
	clFinish(queue);

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	nanoSeconds2 += end - start;
	
	clEnqueueReadBuffer(queue, d_unew, CL_TRUE, 0, sz, unew, 0, 0, 0);
	clEnqueueReadBuffer(queue, d_u, CL_TRUE, 0, sz, u, 0, 0, 0);
	clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0, sz, res, 0, 0, 0);
	
	obj = 0.f;
	for(int j = 0; j < N+2; ++j){
		obj += res[j];
	}
	
	/*if(!(iter%10000)){
		printf("Iter: %d, error = %lg\n", iter, sqrt(obj));
	}*/

	clEnqueueWriteBuffer(queue, d_u, CL_TRUE, 0, sz, unew, 0, 0, 0);
	iter++;
  }

  printf("kernel Jacobi execution time is: %0.3f milliseconds \n",nanoSeconds1 / 1e6);
  printf("kernel reduce execution time is: %0.3f milliseconds \n",nanoSeconds2 / 1e6);
  printf("Final Iteration: %d, obj= %lg\n", iter, sqrt(obj));
  if(plat==0){
	  stop=clock();
	  printf("Total computing time CPU is: %12.11f(s)\n",((float)(stop-begin))/CLOCKS_PER_SEC);
  }
  // blocking read to host 

  float maxerr=0.0;
	for(int ii=0; ii<N*N; ii++){
		maxerr = MAX(maxerr,fabs(u[ii]-b[ii]/(h*h*2.0*M_PI*M_PI)));
	}
	printf("Max error: %lg\n", maxerr);

	// free memory on host
	free(u);
	free(unew);
	free(b);
	free(res);
  /* print out results */
  
}
