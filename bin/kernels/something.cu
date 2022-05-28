extern "C" __global__ void hoge( float* a, float *b ) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float bs[4];
    for( int i = 0 ; i < 4 ; ++i )
    {
        bs[i] = b[i];
    }
    for( int i = 0 ; i < 4 ; ++i )
    {
        a[idx] += bs[i];
    }
}