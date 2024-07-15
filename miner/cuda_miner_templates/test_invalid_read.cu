extern "C" __global__ void kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        printf("hello world\n%d", *((int*)303030304));
    }
}