/*************************************************************************
	> File Name: omp_GaussSeidel.c
	> Author: Ao Cai
	> Mail: aocai166@gmail.com 
	> Created Time: Mon 04 March 2019 01:48:38 PM CST
 ************************************************************************/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#include<omp.h>

double compute_xk(double *u, double *b, int N, double weight, int size)
/* Computing the new xk and send it to u */
{
	int ix, iz, ii;
	double invD, Uu, Lu, diff, tmp, obj=0.0;
	invD = 1/4.0;

#pragma omp parallel for num_threads(size) reduction(+:obj) \
	private(ix, iz, ii, diff, tmp) \
	shared(N, u, b)
	for(iz = 0; iz < N; iz++){
		for(ix = 0; ix < N; ix++){
			ii = ix+iz*N;
			diff = 4.0*u[ii];
			diff -= (iz+1<N)?u[ii+N]:0.0;
			diff -= (ix+1<N)?u[ii+1]:0.0;
			diff -= (iz-1>=0)?u[ii-N]:0.0;
			diff -= (ix-1>=0)?u[ii-1]:0.0;
			tmp = b[ii] - diff;
			obj += tmp*tmp;
		}
	}
#pragma omp parallel for num_threads(size) \
	private(ix, iz, ii, Uu, Lu, tmp) \
	shared(N, u, b, invD)
	for(iz = 0; iz < N; iz++){
		for(ix = 0; ix < N; ix++){
			ii = ix+iz*N;
			Uu = 0.0;
			Uu -= (iz+1<N)?u[ii+N]:0.0;
			Uu -= (ix+1<N)?u[ii+1]:0.0;
			tmp = b[ii] - Uu;
			Lu = 0.0;
			Lu -= (iz-1>=0)?u[ii-N]:0.0;
			Lu -= (ix-1>=0)?u[ii-1]:0.0;
			tmp = tmp - Lu;
			u[ii] = invD*tmp;
		}
	}
	return sqrtf(obj);
}

int main(int argc, char **argv){
	int N = atoi(argv[1]);
	//int N = 100;
	int size = atoi(argv[argc-1]);

	int ii, ix, iz, iter, niter;
	double h, tmp, tmpx, tmpz, obj, objm, weight, tol; // objective function & model difference objective
	double *u, *b; // A is the differential matrix and b is the source function
	double start, stop;

	printf("N=%d\n",N);
	printf("Threads=%d\n",size);
	u = (double*) calloc(N*N, sizeof(double));
	b = (double*) calloc(N*N, sizeof(double));

	memset(u, 1.0, N*N*sizeof(double));

	h = 2.0/(N+1.0);
	iter = 0;
	niter = 100000;
	obj=1.0;
	weight = 0.5;
	tol = 1e-6;
	tmp = h*h;
	start=omp_get_wtime(); // record starting time
	
	omp_set_nested(1);

#pragma omp parallel for num_threads(size) \
	private(iz, ix, ii, tmpx, tmpz) \
	shared(N, h, b)
	for(iz = 0; iz < N; iz++){
		for (ix = 0; ix < N; ix++){
			ii = ix + iz*N;
			tmpx = (ix+1.0)*h-1.0;
			tmpz = (iz+1.0)*h-1.0;
			b[ii] = tmp*sin(M_PI*tmpx)*sin(M_PI*tmpz);
		}
	}
	while (obj>=tol && iter < niter){
		obj = compute_xk(u, b, N, weight, size);
		if((iter%1000)==0 && iter<3001){
			printf("The objective at iter %d is: %12.11f\n",iter,obj);
		}
		iter = iter+1;
	}
	stop=omp_get_wtime(); // record stopping time
	printf("The final objective at iter %d is: %12.11f\n",iter-1,obj);
	if(N==2)printf("The numerical solution u1=%f u2=%f u3=%f u4=%f\n",u[0],u[1],u[2],u[3]);
	printf("The Total computing time is: %g(s)\n",(stop-start));
	
	free(u);
	free(b);
}
