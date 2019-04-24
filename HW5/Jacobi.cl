
__kernel void Jacobi(int N,
		__global float *__restrict__ d_u, 
		__global float *__restrict__ d_b, 
		__global float *__restrict__ d_unew){

	const int id = get_local_id(0) + get_local_size(0)*get_group_id(0);
	const int ix = get_group_id(0);
	const int iz = get_local_id(0);

	float newu = 0.0f;
	const int Np = N+2;

	if(ix > 0 && ix < N+1){
		if(iz > 0 && iz < N+1){
		float ru = -d_u[id-Np]-d_u[id+Np]-d_u[id-1]-d_u[id+1];
	    newu = .25 * (d_b[id] - ru);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(ix > 0 && ix < N+1){
		if(iz > 0 && iz < N+1){
			d_unew[id] = newu;
//		printf("newu = %f\n",newu);
		}
	}
}

