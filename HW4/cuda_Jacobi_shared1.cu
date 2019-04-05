/*************************************************************************
	> File Name: cuda_Jacobi_shared.cu
	> Author: Ao Cai
	> Mail: aocai166@gmail.com 
	> Created Time: April 04 2019 09:48:38 AM CST
 ************************************************************************/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>

#define p_N 100
#define p_Nthreads 256
#define dfloat float  // switch between double/single precision
#define MAX(a,b) ((a)>(b)?(a):(b))

__global__ void compute_xk(int N, dfloat *u, dfloat *b, dfloat *res)
/* Computing the new xk and send it to u */
{
	__shared__ dfloat s_x[3*p_Nthreads];
	__shared__ dfloat s_b[p_Nthreads];

	const int ii = blockIdx.x*blockDim.x + threadIdx.x;
	const int ix = blockIdx.x;
	const int iz = threadIdx.x;

	dfloat newu = 0.0;

	if(ix>0 && ix<N){
		s_x[iz] = u[ii-N];
		s_x[iz+N] = u[ii];
		s_x[iz+2*N] = u[ii+N];
		s_b[iz] = b[ii];
	}

	__syncthreads();

	if(ix > 0 && ix < N){
		if(iz > 0 && iz < N){

			dfloat invD=0.25, Ru, tmp;
			Ru = 0.0;
			Ru -= (iz-1>=0)?s_x[iz]:0.0;
			Ru -= (iz+1<N)?s_x[iz+2*N]:0.0;
			Ru -= (ix-1>=0)?s_x[iz+N-1]:0.0;
			Ru -= (ix+1<N)?s_x[iz+N+1]:0.0;

			tmp = s_b[iz] - Ru;
			newu = invD*tmp;
			tmp = tmp - 4.0*u[ii];
			res[ii] = tmp*tmp;
		}
	}

	__syncthreads();

	if(ix > 0 && ix < N){
		if(iz > 0 && iz < N){
			u[ii] = newu;
		}
	}
}

__global__ void reduce1(int N, float *x, float *xout){

  __shared__ float s_x[p_Nthreads];

  const int tid = threadIdx.x;
  const int i = blockIdx.x*blockDim.x + tid;

  // load smem
  s_x[tid] = 0;
  if (i < N){
    s_x[tid] = x[i];
  }
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2){
    int index = 2*s*tid;
    if (index < blockDim.x){
      s_x[index] += s_x[index+s]; // bank conflicts
    }
    __syncthreads();
  }   

  if (tid==0){
    xout[blockIdx.x] = s_x[0];
  }
}

int main(void){
	//int N = atoi(argv[1]);
	int N = p_N;

	int ii, ix, iz;
	dfloat h, tmp, tmpx, tmpz, obj=1.0, tol; // objective function & model difference objective
	dfloat *u, *b, *res; // A is the differential matrix and b is the source function

	printf("N=%d, thread-block size: %d\n",N, p_Nthreads);
	u = (dfloat*) calloc(N*N, sizeof(dfloat));
	b = (dfloat*) calloc(N*N, sizeof(dfloat));
	res = (dfloat*) calloc(N*N, sizeof(dfloat));

//	for(ii=0; ii<N*N; ii++){
//		u[ii] = 1.0;
//	}

	h = 2.0/(N+1.0);
	tmp = h*h;
	tol = 1e-6;

	for(iz = 0; iz < N; iz++){
		for (ix = 0; ix < N; ix++){
			ii = ix + iz*N;

			tmpx = (ix+1.0)*h-1.0;
			tmpz = (iz+1.0)*h-1.0;
			b[ii] = tmp*sin(M_PI*tmpx)*sin(M_PI*tmpz);
		}
	}

	// Allocate CUDA memory
	dfloat *c_u, *c_b, *c_res, *c_out;
	cudaMalloc(&c_u, N*N*sizeof(dfloat));
	cudaMalloc(&c_b, N*N*sizeof(dfloat));
	cudaMalloc(&c_res, N*N*sizeof(dfloat));

	// Copy host memory over to GPU
	cudaMemcpy(c_u, u, N*N*sizeof(dfloat), cudaMemcpyHostToDevice);
	cudaMemcpy(c_b, b, N*N*sizeof(dfloat), cudaMemcpyHostToDevice);
	cudaMemcpy(c_res, res, N*N*sizeof(dfloat), cudaMemcpyHostToDevice);

	// Initialization
	int Nthreads = N;
	int Nblocks = N;
	dim3 threadsPerBlock(Nthreads,1,1);
	dim3 blocks(Nblocks,1,1);

	int Nthreads_reduce = p_Nthreads;
	int Nblocks_reduce = (N*N+Nthreads_reduce-1)/Nthreads_reduce;
	dim3 threadsPerBlock_reduce(Nthreads_reduce,1,1);
	dim3 blocks_reduce(Nblocks_reduce,1,1);

	dfloat *out = (dfloat*)malloc(Nblocks_reduce*sizeof(dfloat));
	cudaMalloc(&c_out, Nblocks_reduce*sizeof(dfloat));

	int iter = 0;

	while(obj > tol*tol){
		obj = 0.0;

		compute_xk<<< blocks, threadsPerBlock >>> (N, c_u, c_b, c_res);

		reduce1<<< blocks_reduce, threadsPerBlock_reduce >>> (N*N, c_res, c_out);

		cudaMemcpy(out, c_out, Nblocks_reduce*sizeof(dfloat), cudaMemcpyDeviceToHost);

		for (ii=0; ii< Nblocks_reduce; ii++){
			obj += out[ii];
		}
		if(!(iter%1000)){
			printf("Iter: %d, error = %lg\n", iter, sqrt(obj));
		}

		iter++;
	}
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	if(N==2)printf("The numerical solution u1=%f u2=%f u3=%f u4=%f\n",u[0],u[1],u[2],u[3]);

	// check result
	dfloat err=0.0;
	for(int ii=0; ii<N*N; ii++){
		err = MAX(err,fabs(u[ii]-b[ii]/(h*h*2.0*M_PI*M_PI)));
	}
	printf("Final Iteration: %d, obj= %lg\n", iter, sqrt(obj));
	printf("Max error: %lg\n", err);

	// free memory on both CPU and GPU
	cudaFree(c_u);
	cudaFree(c_b);
	cudaFree(c_res);
	cudaFree(c_out);
	free(u);
	free(b);
	free(res);
	free(out);
}
