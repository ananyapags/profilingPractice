#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//one kernal -> exectued on GPu
__global__ void test01(){
 //print block and thread IDs
 printf("The block ID is %d ---The thread ID is %d\n",blockIdx.x, threadIdx.x );
}

//one function ->     exectued on GPU
int main()
{
    //call for the the testo1 kernel

    //kernel_name <<<num of blocks, num of threads per block>>>();
    //1st sm receive the first block
    test01<<<2,8>>>(); 
    cudaDeviceSynchronize();

    //need to return a int bc int fucntion
    return 0;
}

