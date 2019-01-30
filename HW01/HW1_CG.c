/*************************************************************************
	> File Name: HW1_CG.c
	> Conjugate Gradient method
	> Author: Ao Cai
 ************************************************************************/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

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
double compute_alpha(double *temp, double *r, double *p, int N)
/* Computing the parameter alpha, temp is the working matrix */
{
	int ii;
	double alpha, alpha1, alpha2;
	alpha1 = alpha2 = 0.0;
	compute_Au(temp, p, N);
	for(ii = 0; ii < N*N; ii++){
		alpha1 += r[ii]*r[ii];
		alpha2 += p[ii]*temp[ii];
	}
	alpha = alpha1/alpha2;
	return alpha;
}
void compute_xk(double *u, double *p, double alpha, int N)
/* Computing the new xk and send it to u using conjugate gradient method */
{
	int ii;

	for(ii = 0; ii < N*N; ii++){
			u[ii] = u[ii] + alpha*p[ii];
	}
}
double compute_rk(double *temp, double *r, double alpha, int N)
/* Updating the residual rk, temp is the working matrix */
{
	int ii;
	double norm, tmp;
	norm = 0.0;
	for(ii = 0; ii < N*N; ii++){
		tmp = r[ii];
		norm += tmp*tmp;
		r[ii] = r[ii] - alpha*temp[ii];
	}
	return norm;
}
double compute_beta(double *r, double norm, int N)
/* Computing the new pk for conjugate gradient method */
{
	int ii;
	double beta, beta0, tmp;
	beta0 = 0.0;

	for(ii = 0; ii < N*N; ii++){
		tmp = r[ii];
		beta0 += tmp*tmp;
	}
	beta = beta0/norm;
	return beta;
}
void compute_pk(double *r, double *p, double beta, int N)
/* Computing the new pk for conjugate gradient method */
{
	int ii;

	for(ii = 0; ii < N*N; ii++){
			p[ii] = r[ii] + beta*p[ii];
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

int main(int argc, char **argv){
	int N = 20000;
	int ii, ix, iz, iter, niter;
	double h, tmp, tmpx, tmpz, obj, objm, tol, alpha, beta, norm; // objective function & model difference objective
	double *u, *u0=NULL, *r, *p, *b, *temp; // A is the differential matrix and b is the source function
	clock_t start=0, stop; /* timer */

	u = (double*) calloc(N*N, sizeof(double));
	r = (double*) calloc(N*N, sizeof(double));
	p = (double*) calloc(N*N, sizeof(double));
	b = (double*) calloc(N*N, sizeof(double));
	temp = (double*) calloc(N*N, sizeof(double));
	u0 = (double*) calloc(N*N, sizeof(double));

	memset(u, 1.0, N*N*sizeof(double));

	h = 2.0/(N+1.0);
	iter = 0;
	niter = 10000;
	obj = 1.0;
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
	/* Initialize the optimization */
	compute_Au(temp, u, N);
	for(ii = 0; ii < N*N; ii++){
		tmp = b[ii] - temp[ii];
		r[ii] = tmp;
		p[ii] = tmp;
	}
	/* Applying conjugate gradient method */
	while (obj >= tol && iter < niter){
		memset(temp, 0.0, N*N*sizeof(double));
		compute_Au(temp, u, N);
		obj = compute_obj(temp, b, N*N);
		objm = compute_obj(u, u0, N*N); /* L2-norm objective function for model */
		memset(temp, 0.0, N*N*sizeof(double));
		alpha = compute_alpha(temp, r, p, N);
		compute_xk(u, p, alpha, N);
		norm = compute_rk(temp, r, alpha, N);
		beta = compute_beta(r, norm, N);
		compute_pk(r, p, norm, N);

		//if((iter%100)==0){
			printf("The objective at iter %d is: %12.11f\n",iter,obj);
			printf("The model objective at iter %d is: %12.11f\n",iter,objm);
		//}
		iter = iter+1;
	}
	stop=clock(); // record stopping time
	printf("The final objective at iter %d is: %12.11f\n",iter-1,obj);
	printf("The final model objective at iter %d is: %12.11f\n",iter-1,objm);
	printf("The Total computing time is: %f(s)\n",((float)(stop-start))/CLOCKS_PER_SEC);
	
	free(u);
	free(r);
	free(p);
	free(u0);
	free(b);
	free(temp);
}
