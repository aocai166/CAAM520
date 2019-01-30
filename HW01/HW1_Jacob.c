/*************************************************************************
	> File Name: HW1_Jacob.c
	> Author: Ao Cai
	> Mail: aocai166@gmail.com 
	> Created Time: Sun 27 Jan 2019 01:48:38 PM CST
 ************************************************************************/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

void compute_Ru(double *temp, double *u, int N)
/* Computing the matrix product Ru */
{
	int ix,iz,ii;
	double diff1, diff2;
	for(iz=0; iz < N; iz++){
		for(ix=0; ix < N; ix++){
			ii = ix+iz*N;
			diff1 = diff2 = 0.0;
			diff1 -= (iz-1>=0)?u[ii-N]:0.0;
			diff1 -= (iz+1<N)?u[ii+N]:0.0;
			diff2 -= (ix-1>=0)?u[ii-1]:0.0;
			diff2 -= (ix+1<N)?u[ii+1]:0.0;
			temp[ii] = diff1+diff2;
		}
	}
}
void compute_Au(double *temp, double *u, int N)
/* Computing the matrix product Ru */
{
	int ix,iz,ii;
	double diff1, diff2;
	for(iz=0; iz < N; iz++){
		for(ix=0; ix < N; ix++){
			ii = ix+iz*N;
			diff1 = diff2 = 2.0*u[ii];
			diff1 -= (iz-1>=0)?u[ii-N]:0.0;
			diff1 -= (iz+1<N)?u[ii+N]:0.0;
			diff2 -= (ix-1>=0)?u[ii-1]:0.0;
			diff2 -= (ix+1<N)?u[ii+1]:0.0;
			temp[ii] = diff1+diff2;
		}
	}
}
double compute_obj(double *dcal, double *dobs, int N)
{
	int ii;
	double tmp, obj = 0.0;
	for (ii = 0; ii < N; ii++){
		tmp = dcal[ii]-dobs[ii];
		obj += tmp*tmp;
	}
	return sqrtf(obj);
}
void compute_xk(double *u, double *temp, double *b, int N, double weight)
/* Computing the new xk and send it to u */
{
	int ix, iz, ii;
	double tmp;

	tmp = 1/4.0;
	for(iz = 0; iz < N; iz++){
		for(ix = 0; ix < N; ix++){
			ii = ix+iz*N;
			u[ii] = weight*tmp*(b[ii]-temp[ii])+ (1.0-weight)*u[ii];
		}
	}
}

int main(int argc, char **argv){
	int N = 100;
	int ii, ix, iz, iter, niter;
	double h, tmp, tmpx, tmpz, obj, objm, weight, tol; // objective function & model difference objective
	double *u, *u0, *b, *temp; // A is the differential matrix and b is the source function
	clock_t start=0, stop; /* timer */

	u = (double*) calloc(N*N, sizeof(double));
	b = (double*) calloc(N*N, sizeof(double));
	temp = (double*) calloc(N*N, sizeof(double));
	u0 = (double*) calloc(N*N, sizeof(double));

	memset(u, 1.0, N*N*sizeof(double));

	h = 2.0/(N+1.0);
	iter = 0;
	niter = 100000;
	obj=1.0;
	weight = 0.5;
	tol = 1e-6;
	tmp = h*h;
	start=clock(); // record starting time
	
	/* build the f function f = sin(pi*x)sin(pi*y) */
	for(iz = 0; iz < N; iz++){
		for (ix = 0; ix < N; ix++){
			ii = ix + iz*N;
			tmpx = (ix+1.0)*h-1.0;
			tmpz = (iz+1.0)*h-1.0;
			b[ii] = tmp*sin(M_PI*tmpx)*sin(M_PI*tmpz);
			u0[ii] = sin(M_PI*tmpx)*sin(M_PI*tmpz)/2.0/M_PI/M_PI;
		}
	}
	while (obj>=tol && iter < niter){
		compute_Au(temp, u, N);
		obj = compute_obj(temp, b, N*N); /* L2-norm objective function for residuals */
		objm = compute_obj(u, u0, N*N); /* L2-norm objective function for model */
		memset(temp, 0.0, N*N*sizeof(double));
		compute_Ru(temp, u, N);
		compute_xk(u, temp, b, N, weight);
		if((iter%100)==0){
			printf("The objective at iter %d is: %12.11f\n",iter,obj);
			printf("The model objective at iter %d is: %12.11f\n",iter,objm);
		}
		iter = iter+1;
	}
	stop=clock(); // record stopping time
	printf("The final objective at iter %d is: %12.11f\n",iter-1,obj);
	printf("The final model objective at iter %d is: %12.11f\n",iter-1,objm);
	printf("The Total computing time is: %f(s)\n",((float)(stop-start))/CLOCKS_PER_SEC);
	
	free(u);
	free(u0);
	free(b);
	free(temp);
}
