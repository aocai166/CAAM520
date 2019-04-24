
__kernel void reduce(int N,
		__global const float *__restrict__ d_u,
		__global const float *__restrict__ d_unew,
		__global float *__restrict__ d_res){

	__local float s_res[BDIM+2];

	const int ii = get_local_id(0) + get_local_size(0)*get_group_id(0);
	const int tid = get_local_id(0);
	const int Nblock = get_local_size(0);
	const int bid = get_group_id(0);

	s_res[tid]=0.f;
	if(ii < N){
		const float diff = d_unew[ii] - d_u[ii];
		s_res[tid] = diff*diff;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	for(unsigned int s = Nblock/2; s > 0; s/=2){
		if(tid < Nblock){
			s_res[tid] += s_res[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(tid==0){
		d_res[bid] = s_res[0];
	}
}
