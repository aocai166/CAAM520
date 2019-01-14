#include <stdio.h>
#include <stdlib.h>
//Ping-pong with maxping=10;
// include headers for MPI definitions
#include <mpi.h>

// purpose: demonstrate 
//          1. sending a message from process 0 to process 2
//          2. receiving a message by process 2 from process 0
// 
//          using MPI functions MPI_Send and 

// main program - we need those input args
int main(int argc, char **argv){

  // This code gets executed by all processes launched by mpiexec.

  // variable to store the MPI rank of this process
  // rank signifies a unique integer identifier (0-indexed)
  int rank, ping, maxping=10;

  // start MPI environment
  MPI_Init(&argc, &argv);

  // find rank (number of this process)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //MPI_Request req, rreq;

 	MPI_Status status;
	int N = 1;
	int tag = 999;
	int dest = (rank+1) % 2;
	//int *ping = (int*) calloc(N, sizeof(int));
	int *recvdata = (int*) calloc(N, sizeof(int));
	ping=0;
	//senddata[0]=ping;
	recvdata[0]=0;

	while(ping<maxping){
	 // send the message if you are process 0
 	 if(rank== ping % 2){
 	   // message

	   ping++;
 	   // send message 
 	   MPI_Send(&ping, N, MPI_INT, dest, tag, MPI_COMM_WORLD);
 	   //MPI_Irecv(ping, N, MPI_INT, destination, tag, MPI_COMM_WORLD,&rreq);

 	   printf("Process %d sent and incremented pingpong: %d to Processor %d\n", rank ,ping, dest);
 	 }else{
 	 // receive the message if you are process 1
 	   // message

 	   // receive message 
 	   MPI_Recv(&ping, N, MPI_INT, dest, tag, MPI_COMM_WORLD, &status);
	   //ping=ping+1;
 	   //MPI_Isend(ping, N, MPI_INT, origin, tag, MPI_COMM_WORLD,&req);

 	   printf("Process %d received pingpong: %d from Processor %d\n", rank, ping, dest);
 	 }
	 //MPI_Barrier(MPI_COMM_WORLD);
	}

  // close down MPI environment
  MPI_Finalize();

}

