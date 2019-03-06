/*************************************************************************
	> File Name: HW1_CG.c
	> Conjugate Gradient method parallel by openmp
	> Author: Ao Cai
 ************************************************************************/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#include<omp.h>

void compute_Au(double *temp, double *u, int N, int size)
/* Computing the matrix product Ru */
{
	int ix,iz,ii;
	double diff1, diff2;

#pragma omp parallel for num_threads(size) \
	private(iz, ix, ii, diff1, diff2) \
	shared(N, u, temp)
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
double compute_alpha(double *temp, double *r, double *p, int N, int size)
/* Computing the parameter alpha, temp is the working matrix */
{
	int ii;
	double alpha, alpha1, alpha2;
	alpha1 = alpha2 = 0.0;
	compute_Au(temp, p, N, size);

#pragma omp parallel for num_threads(size) reduction(+:alpha1) reduction(+:alpha2) \
	private(ii) shared(r, p, temp, N)
	for(ii = 0; ii < N*N; ii++){
		alpha1 += r[ii]*r[ii];
		alpha2 += p[ii]*temp[ii];
	}
	alpha = alpha1/alpha2;
	return alpha;
}
void compute_xk(double *u, double *p, double alpha, int N, int size)
/* Computing the new xk and send it to u using conjugate gradient method */
{
	int ii;

#pragma omp parallel for num_threads(size) \
	private(ii) shared(u, alpha, p, N)
	for(ii = 0; ii < N*N; ii++){
			u[ii] = u[ii] + alpha*p[ii];
	}
}
double compute_rk(double *temp, double *r, double alpha, int N, int size)
/* Updating the residual rk, temp is the working matrix */
{
	int ii;
	double norm, tmp;
	norm = 0.0;

#pragma omp parallel for num_threads(size)  reduction(+:norm) \
	private(ii, tmp) shared(r, alpha, temp, N)
	for(ii = 0; ii < N*N; ii++){
		tmp = r[ii];
		norm += tmp*tmp;
		r[ii] = r[ii] - alpha*temp[ii];
	}
	return norm;
}
double compute_beta(double *r, double norm, int N, int size)
/* Computing the new pk for conjugate gradient method */
{
	int ii;
	double beta, beta0, tmp;
	beta0 = 0.0;

#pragma omp parallel for num_threads(size)  reduction(+:beta0) \
	private(ii, tmp) shared(r, N)
	for(ii = 0; ii < N*N; ii++){
		tmp = r[ii];
		beta0 += tmp*tmp;
	}
	beta = beta0/norm;
	return beta;
}
void compute_pk(double *r, double *p, double beta, int N, int size)
/* Computing the new pk for conjugate gradient method */
{
	int ii;

#pragma omp parallel for num_threads(size) \
	private(ii) shared(r, p, beta, N)
	for(ii = 0; ii < N*N; ii++){
			p[ii] = r[ii] + beta*p[ii];
	}
}
double compute_obj(double *dcal, double *dobs, int N, int size)
{
	int ii;
	double tmp, obj = 0.0;

#pragma omp parallel for num_threads(size) reduction(+:obj) \
	private(ii, tmp) shared(dcal, dobs, N)
	for (ii = 0; ii < N; ii++){
		tmp = dcal[ii]-dobs[ii];
		obj += tmp*tmp;
	}
	return sqrtf(obj);
}

int main(int argc, char **argv){
	int N = atoi(argv[1]);
	// int N = 1000;
	int size = atoi(argv[argc-1]);

	int ii, ix, iz, iter, niter;
	double h, tmp, tmpx, tmpz, obj, tol, alpha, beta, norm; // objective function & model difference objective
	double *u, *r, *p, *b, *temp; // A is the differential matrix and b is the source function
	double start, stop; /* timer */

	printf("N=%d\n",N);
	printf("Thread=%d\n",size);

	u = (double*) calloc(N*N, sizeof(double));
	r = (double*) calloc(N*N, sizeof(double));
	p = (double*) calloc(N*N, sizeof(double));
	b = (double*) calloc(N*N, sizeof(double));
	temp = (double*) calloc(N*N, sizeof(double));

	memset(u, 1.0, N*N*sizeof(double));

	h = 2.0/(N+1.0);
	iter = 0;
	niter = 10000;
	obj = 1.0;
	tol = 1e-6;
	tmp = h*h;
	start=omp_get_wtime(); // record starting time
	
	omp_set_nested(1);
	/* build the f function f = sin(pi*x)sin(pi*y) */
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
	/* Initialize the optimization */
	compute_Au(temp, u, N, size);
	for(ii = 0; ii < N*N; ii++){
		tmp = b[ii] - temp[ii];
		r[ii] = tmp;
		p[ii] = tmp;
	}
	/* Applying conjugate gradient method */
	while (obj >= tol && iter < niter){
		memset(temp, 0.0, N*N*sizeof(double));
		compute_Au(temp, u, N, size);
		obj = compute_obj(temp, b, N*N, size);
		memset(temp, 0.0, N*N*sizeof(double));
		alpha = compute_alpha(temp, r, p, N, size);
		compute_xk(u, p, alpha, N, size);
		norm = compute_rk(temp, r, alpha, N, size);
		beta = compute_beta(r, norm, N, size);
		compute_pk(r, p, norm, N, size);

		//if((iter%100)==0){
			printf("The objective at iter %d is: %12.11f\n",iter,obj);
		//}
		iter = iter+1;
	}
	stop=omp_get_wtime(); // record stopping time
	printf("The final objective at iter %d is: %12.11f\n",iter-1,obj);
	printf("The Total computing time is: %g(s)\n",(stop-start));
	
	free(u);
	free(r);
	free(p);
	free(b);
	free(temp);
}
