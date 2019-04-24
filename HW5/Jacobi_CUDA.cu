/* Modified from Jessie's reference code */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265359f
#define MAX(a,b) (((a)>(b))?(a):(b))
#define p_Nthreads 32

__global__ void jacobi(int N, float * u, float *f, float *unew){
  
  const int i = threadIdx.x + blockIdx.x*blockDim.x + 1; // offset by 1
  const int j = threadIdx.y + blockIdx.y*blockDim.y + 1;

  if (i < N+1 && j < N+1){
    const int Np = (N+2);
    const int id = i + j*(N+2);
    const float ru = -u[id-Np]-u[id+Np]-u[id-1]-u[id+1];
    const float newu = .25 * (f[id] - ru);
    unew[id] = newu;
  }
}

// use all threads
__global__ void reduce(int N2, float *u, float *unew, float *res){

  __shared__ volatile float s_x[p_Nthreads]; // volatile for in-warp smem mods

  const int tid = threadIdx.x;
  const int i = tid + blockIdx.x*(2*blockDim.x);

  s_x[tid] = 0;
  if (i < N2){
    const float unew1 = unew[i];
    const float unew2 = unew[i + blockDim.x];
    const float diff1 = unew1 - u[i];
    const float diff2 = unew2 - u[i + blockDim.x];
    s_x[tid] = diff1*diff1 + diff2*diff2; 

    // update u
    u[i] = unew1;
    u[i + blockDim.x] = unew2;
  }
  __syncthreads();
  
  // stop at s = 64
  for (unsigned int s = blockDim.x/2; s > 32; s /= 2){
    if (tid < s){
      s_x[tid] += s_x[tid+s]; 
    }
    __syncthreads();
  }   

  // manually reduce within a warp
  if (tid < 32){
    s_x[tid] += s_x[tid + 32];
    s_x[tid] += s_x[tid + 16];
    s_x[tid] += s_x[tid + 8];
    s_x[tid] += s_x[tid + 4];
    s_x[tid] += s_x[tid + 2];
    s_x[tid] += s_x[tid + 1];   
  }
  if (tid==0){
    res[blockIdx.x] = s_x[0];
  }
}


int main(int argc, char **argv){
   
  int N = atoi(argv[1]);
  float tol = atof(argv[2]);

  float *u = (float*) calloc((N+2)*(N+2), sizeof(float));
  float *unew = (float*)calloc((N+2)*(N+2),sizeof(float));
  float *f = (float*) calloc((N+2)*(N+2), sizeof(float));
  float h = 2.0/(N+1);
  for (int i = 0; i < N+2; ++i){
    for (int j = 0; j < N+2; ++j){
      const float x = -1.0 + i*h;
      const float y = -1.0 + j*h;
      f[i + j*(N+2)] = sin(PI*x)*sin(PI*y) * h*h;
    }
  } 

  // cuda memory for Jacobi variables
  float *c_u, *c_f, *c_unew;
  cudaMalloc(&c_u, (N+2)*(N+2)*sizeof(float));
  cudaMalloc(&c_f, (N+2)*(N+2)*sizeof(float));
  cudaMalloc(&c_unew, (N+2)*(N+2)*sizeof(float));
  cudaMemcpy(c_u,u, (N+2)*(N+2)*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(c_f,f, (N+2)*(N+2)*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(c_unew,unew,(N+2)*(N+2)*sizeof(float),cudaMemcpyHostToDevice);

  // run kernel, copy result back to CPU
  int Nthreads = p_Nthreads; // good if it's a multiple of 32, can't have more than 1024
  int Nblocks = (N + Nthreads-1)/Nthreads; 
  dim3 threadsPerBlock(Nthreads,Nthreads,1);  
  dim3 blocks(Nblocks,Nblocks,1);

  // for reduce kernel
  int Nthreads1D = p_Nthreads; 
  int Nblocks1D = ((N+2)*(N+2) + Nthreads-1)/Nthreads; 
  int halfNblocks1D = (Nblocks1D + 1)/2; 
  dim3 threadsPerBlock1D(Nthreads1D,1,1);  
  dim3 halfblocks1D(halfNblocks1D,1,1);

  // storage for residual
  float *res = (float*) calloc(halfNblocks1D, sizeof(float));
  float *c_res;
  cudaMalloc(&c_res, halfNblocks1D*sizeof(float));

  float time, time1=0.f, time2=0.f;
  cudaEvent_t start, stop;
  int iter = 0;
  float r2 = 1.;
  while (r2 > tol*tol){

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
    jacobi <<< blocks, threadsPerBlock >>> (N, c_u, c_f, c_unew);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	time1 += time;

	cudaEventRecord(start,0);
    reduce <<< halfblocks1D, threadsPerBlock1D >>> ((N+2)*(N+2), c_u, c_unew, c_res);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	time2 += time;

    // finish block reduction on CPU
    cudaMemcpy(res,c_res,halfNblocks1D*sizeof(float),cudaMemcpyDeviceToHost);
    r2 = 0.f;
    for (int j = 0; j < halfNblocks1D; ++j){
      r2 += res[j];
    }

	if((!iter%1000)){
		printf("Iter: %d. error = %lg\n", iter, sqrt(r2));
	}
    ++iter;
  }
 
  cudaMemcpy(u,c_unew,(N+2)*(N+2)*sizeof(float),cudaMemcpyDeviceToHost);

  float err = 0.0;
  for (int i = 0; i < (N+2)*(N+2); ++i){
    err = MAX(err,fabs(u[i] - f[i]/(h*h*2.0*PI*PI)));
  }
  
  printf("Max error: %f, r2 = %lg, iterations = %d\n", err,sqrt(r2),iter);
  printf("Jacobi kernel computing time is: %f milliseconds\n", time1);
  printf("Reduction kernel computing time is: %f milliseconds\n", time2);

}
