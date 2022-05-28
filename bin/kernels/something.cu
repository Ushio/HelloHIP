extern "C" __global__ void hoge( ) 
{
    printf( "%d-%d / %d\n", blockIdx.x, blockIdx.y, threadIdx.x );
}