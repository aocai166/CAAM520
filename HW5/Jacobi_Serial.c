/* Modified from Jessie's Serial code */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.14159265359
#define MAX(a,b) (((a)>(b))?(a):(b))

// solve for solution vector u
int solve(const int N, const double tol, double * u, double * f){

  double *unew = (double*)calloc((N+2)*(N+2),sizeof(double));
  
  double w = 1.0;
  double invD = 1./4.;  // factor of h cancels out

  double res2 = 1.0;
  unsigned int iter = 0;
  while(res2>tol*tol){

    res2 = 0.0;

    // update interior nodes using Jacobi
    for(int i=1; i<=N; ++i){
      for(int j=1; j<=N; ++j){
	
	const int id = i + j*(N+2); // x-index first
	const double Ru = -u[id-(N+2)]-u[id+(N+2)]-u[id-1]-u[id+1];
	const double rhs = invD*(f[id]-Ru);
	const double oldu = u[id];
	const double newu = w*rhs + (1.0-w)*oldu;
	res2 += (newu-oldu)*(newu-oldu);
	unew[id] = newu;
      }
    }

    for (int i = 0; i < (N+2)*(N+2); ++i){
      u[i] = unew[i];
    }
    
    if(!(iter%1000)){
      printf("at iter %d: residual = %g\n", iter, sqrt(res2));
    }
    ++iter;
  }

  printf("r2 = %lg, iterations = %d\n", sqrt(res2), iter);
  return iter;
}

int main(int argc, char **argv){
  
  if(argc!=3){
    printf("Usage: ./main N tol\n");
    exit(-1);
  }
  
  int N = atoi(argv[1]);
  double tol = atof(argv[2]);

  double *u = (double*) calloc((N+2)*(N+2), sizeof(double));
  double *f = (double*) calloc((N+2)*(N+2), sizeof(double));
  double h = 2.0/(N+1);
  clock_t start = 0, stop; /* timer */

  for (int i = 0; i < N+2; ++i){
    for (int j = 0; j < N+2; ++j){
      const double x = -1.0 + i*h;
      const double y = -1.0 + j*h;
      f[i + j*(N+2)] = sin(PI*x)*sin(PI*y) * h*h;
    }
  }

  start=clock(); 
  int iter = solve(N, tol, u, f);

  stop=clock();
  double err = 0.0;
  for (int i = 0; i < (N+2)*(N+2); ++i){
    err = MAX(err,fabs(u[i] - f[i]/(h*h*2.0*PI*PI)));
  }
  
  printf("Max error: %f\n", err);
  printf("Serial code computing time is: %12.11f(s)\n", ((float)(stop-start))/CLOCKS_PER_SEC);
  
  free(u);
  free(f);  

}
