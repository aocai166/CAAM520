
__kernel void reduce(int N,
		__global const float *__restrict__ d_u,
		__global const float *__restrict__ d_unew,
		__global float *__restrict__ d_res){

	__local float s_x[BDIM+2];

	const int id = get_local_id(0) + get_local_size(0)*get_group_id(0);
	const int tid = get_local_id(0);
	const int Nblock = get_local_size(0);
	const int bid = get_group_id(0);

	s_x[tid]=0.f;
	if(id < N){
		const float diff = d_unew[id] - d_u[id];
		s_x[tid] = diff*diff;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	for(unsigned int s = Nblock/2; s > 0; s/=2){
		if(tid<s){
			s_x[tid] += s_x[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(tid==0){
		d_res[bid] = s_x[0];
	}
}
