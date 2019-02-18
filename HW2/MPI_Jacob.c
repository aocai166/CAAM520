/*************************************************************************
	> File Name: MPI_Jacob.c
	> Author: Ao Cai
	> Mail: aocai166@gmail.com 
	> Created Time: Sun 27 Jan 2019 01:48:38 PM CST
	> Compile: mpicc -lm MPI_Jacob.c -o MPI_Jacob
	> Run: mpiexec -np n ./MPI_Jacob N
 ************************************************************************/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<mpi.h>

double compute_xk(double *u, double *b, int N, int nub, double weight)
/* Computing the new xk and send it to u */
{
	int ix, iz, ii;
	double invD, Ru, rhs, oldu, newu, tmp, localobj=0.0;

	invD = 1/4.0;
	for(iz = 1; iz < nub-1; iz++){
		for(ix = 0; ix < N; ix++){
			ii = ix+iz*N;
			Ru = -u[ii-N]-u[ii+N];
			Ru -= (ix-1>=0)?u[ii-1]:0.0;
			Ru -= (ix+1<N)?u[ii+1]:0.0;
			tmp = b[ii] - Ru;
			rhs = invD*tmp;
			tmp = tmp - 4.0*u[ii];
			localobj += tmp*tmp;
			oldu = u[ii];
			newu = weight*rhs + (1.0-weight)*oldu;
			u[ii] = newu;
		}
	}
	return localobj;
}
void bndr_update(double *u, double *bndr, int nub, int N)
{
	int ix, ii;
	for(ix=0; ix<N; ix++){
		u[ix] = bndr[ix];
		ii = ix + N*(nub-1);
		u[ii] = bndr[ix+N];
	}
}

int main(int argc, char **argv){

	//int N = 500;
	int rank, size, node, nres, nub, src, dest; //memory size allocated at the procs 
	int ii, ix, iz, iter, niter;
	double h, tmp, tmpx, tmpz, obj, localobj=0.0, weight, tol; // objective function & model difference objective
	double *u, *u0, *b, *bndr, *sendbuf, *recvbuf; // A is the differential matrix and b is the source function
	clock_t start=0, stop; /* timer */
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

/*	if(argc!=3){
		printf("Usage: ./main N tol\n");
		exit(-1);
	}*/

	int N = atoi(argv[1]);
	if(rank==0)printf("N=%d\n",N);

	node = N/size;
	nres = N - node*size;
	// allocate different memory based on rank
	if(rank == size-1){
		nub = node+nres+2;
	}else{
		nub = node+2;
	}
	u = (double*) calloc(N*nub, sizeof(double));
	b = (double*) calloc(N*nub, sizeof(double));
	u0 = (double*) calloc(N*nub, sizeof(double));
	bndr = (double*) calloc(2*N, sizeof(double));

	memset(u, 1.0, N*nub*sizeof(double));
	if(rank==0){
		memset(&u[0], 0.0, N*sizeof(double));
	}
	if(rank==size-1){
		memset(&u[N*(nub-1)], 0.0, N*sizeof(double));
	}

	h = 2.0/(N+1.0);
	iter = 0;
	niter = 100000;
	obj=1.0;
	weight = 0.5;
	tol = 1e-6;
	tol = tol*tol;
	tmp = h*h;
	start=clock(); // record starting time

	//printf("The nub, node, nres on processor %d is: %d, %d, %d size=%d\n", rank, nub, node, nres, size);
	
	/* build the f function f = sin(pi*x)sin(pi*y) */
	for(iz = 1; iz < nub-1; iz++){
		for (ix = 0; ix < N; ix++){
			ii = ix + iz*N;
			tmpx = (ix+1.0)*h-1.0;
			tmpz = (iz+rank*node)*h-1.0;
			b[ii] = tmp*sin(M_PI*tmpx)*sin(M_PI*tmpz);
			u0[ii] = sin(M_PI*tmpx)*sin(M_PI*tmpz)/2.0/M_PI/M_PI;
		}
	}
	while (obj >= tol && iter < niter){
		obj = 0.0;
		memset(bndr, 0.0, 2*N*sizeof(double));
		localobj = compute_xk(u, b, N, nub, weight);
		//printf("local obj on porc %d is %12.11f\n", rank, localobj);

		if(size>1){
			MPI_Reduce(&localobj, &obj, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Bcast(&obj, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);

			sendbuf = &u[N]; // u[N] stores the top boundary of the decomposited wavefield
			recvbuf = &bndr[N]; // bndr[N] is the transferring wavefield to update the boundaries in the decomposited wavefield
			if(rank%2==1){
				src = rank+1;
				dest = rank-1;
			if(rank > 0) MPI_Send(sendbuf, N, MPI_DOUBLE, dest, 123, MPI_COMM_WORLD);
			if(rank < size-1) MPI_Recv(recvbuf, N, MPI_DOUBLE, src, 99, MPI_COMM_WORLD, &status);
			}else{
				src = rank+1;
				dest = rank-1;
			if(rank < size-1) MPI_Recv(recvbuf, N, MPI_DOUBLE, src, 123, MPI_COMM_WORLD, &status);
			if(rank > 0) MPI_Send(sendbuf, N, MPI_DOUBLE, dest, 99, MPI_COMM_WORLD);
			}

			sendbuf = &u[N*(nub-2)]; // u[N*(nub-2)] stores the bottom boundary of the decomposited wavefield
			recvbuf = &bndr[0]; // bndr[0] is the transferring wavefield to update the top boundaries in the decomposited wavefield
			if(rank%2==1){
				src = rank-1;
				dest = rank+1;
			if(rank < size-1) MPI_Send(sendbuf, N, MPI_DOUBLE, dest, 123, MPI_COMM_WORLD);
			if(rank > 0) MPI_Recv(recvbuf, N, MPI_DOUBLE, src, 99, MPI_COMM_WORLD, &status);
			}else{
				src = rank-1;
				dest = rank+1;
			if(rank > 0) MPI_Recv(recvbuf, N, MPI_DOUBLE, src, 123, MPI_COMM_WORLD, &status);
			if(rank < size-1) MPI_Send(sendbuf, N, MPI_DOUBLE, dest, 99, MPI_COMM_WORLD);
			}
		}else obj = localobj;

		bndr_update(u, bndr, nub, N);

		if((iter%1000)==0 && rank == 0){
			printf("The objective at iter %d is: %12.11f\n",iter,sqrtf(obj));
		}
		iter = iter+1;
	}

	stop=clock(); // record stopping time
	if(rank==0){
		printf("The final objective at iter %d is: %12.11f\n",iter-1,sqrtf(obj));
	//	if(N==2)printf("The numerical solution u1=%f u2=%f u3=%f u4=%f\n",u[0],u[1],u[2],u[3]);
	//	if(N==2)printf("The b matrix b1=%f b2=%f b3=%f b4=%f\n",b[0],b[1],b[2],b[3]);
		printf("The Total computing time is: %f(s)\n",((float)(stop-start))/CLOCKS_PER_SEC);
	}
	free(u);
	free(u0);
	free(b);
	free(bndr);

	MPI_Finalize();
	return 0;
}
