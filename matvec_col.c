// Column based parallel matrix-vector product
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(int argc, char **argv){

	int rank, size, i, j;
	double *sendbuf, *recvbuf, *b = NULL;
	double x_local=0.0;

	/* initialize MPI */
	MPI_Init(&argc, &argv);

	/* find MPI rank and size */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* initialize x in rank 0 */
	if(rank==0){
		double *x_root = (double*)calloc(size,sizeof(double));
		for (i =0; i<size; i++){
			x_root[i] = (double) i + 1.0;
		}
		sendbuf = x_root;
		recvbuf = &x_local;
	}else{
		sendbuf = NULL;
		recvbuf = &x_local;
	}

	double *ai = (double*) calloc(size,sizeof(double));
	for(j =0; j< size; ++j){
		ai[j] = (double) j + rank + 1.0;
	}

	MPI_Scatter(sendbuf, 1, MPI_DOUBLE, recvbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	printf("x[rank = %d] = %f\n", rank, x_local);

	double *x_global = (double*) calloc(size,sizeof(double));
	double *ax_global = (double*) calloc(size,sizeof(double));
	for(j = 0; j < size; ++j){
		x_global[j] = ai[j]*x_local;
	}
	MPI_Alltoall(x_global, 1, MPI_DOUBLE, ax_global, 1, MPI_DOUBLE, MPI_COMM_WORLD);

	double bi = 0.0;
	for(j = 0; j < size; ++j){
		bi += ax_global[j];
	}

	if(rank == 0){
		b = (double*)calloc(size,sizeof(double));
		sendbuf = &bi;
		recvbuf = b;
	}else{
		sendbuf = &bi;
		recvbuf = NULL;
	}
		
	MPI_Gather(sendbuf, 1, MPI_DOUBLE, recvbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(rank == 0){
		for(j = 0; j < size; ++j){
			printf("b[%d] = %f\n", j, b[j]);
		}
	}
	//if(rank==0){ free(x_root);}
	free(ai);
	free(x_global);
	free(ax_global);
	MPI_Finalize();
}
