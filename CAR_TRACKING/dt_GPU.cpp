#include <stdio.h>
#include <stdlib.h>


#include "for_use_GPU.h"

#define s_free(a) {free(a);a=NULL;}

FLOAT *dt_GPU(FLOAT *vals,FLOAT ax,FLOAT bx,FLOAT ay,FLOAT by,int *dims,int *Ix,int *Iy)
{
  const int SQ = dims[0]*dims[1];
  FLOAT *M = (FLOAT*)malloc(sizeof(FLOAT)*SQ);
  FLOAT *tmpM = (FLOAT*)malloc(sizeof(FLOAT)*SQ);
  int *tmpIx = (int*)malloc(sizeof(int)*SQ);
  int *tmpIy = (int*)malloc(sizeof(int)*SQ);
  //  int XD=0;

  CUresult res;
  CUdeviceptr vals_dev, M_dev, tmpM_dev, tmpIx_dev, tmpIy_dev;
  int thread_num = 0, block_num = 0;


  // /* define CUDA block shape */
  // int max_threads_num = 0;
  // res = cuDeviceGetAttribute(&max_threads_num, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
  // if(res != CUDA_SUCCESS){
  //   printf("\ncuDeviceGetAttribute() failed: res = %s\n", conv(res));
  //   exit(1);
  // }
  
  // NR_MAXTHREADS_X = (int)sqrt((double)max_threads_num);
  // NR_MAXTHREADS_Y = (int)sqrt((double)max_threads_num);
  
  // thread_num_x = (dims[1] < NR_MAXTHREADS_X) ? dims[1] : NR_MAXTHREADS_X;
  // thread_num_y = (dims[0] < NR_MAXTHREADS_Y) ? dims[0] : NR_MAXTHREADS_Y;
  
  // block_num_x = dims[1] / thread_num_x;
  // block_num_y = dims[0] / thread_num_y;
  // if(dims[1] % thread_num_x != 0) block_num_x++;
  // if(dims[0] % thread_num_y != 0) block_num_y++;
  

  /* allocate GPU memory */
  res = cuMemAlloc(&vals_dev, SQ*sizeof(FLOAT));
  if(res != CUDA_SUCCESS) {
    printf("cuMemAlloc(vals_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemAlloc(&M_dev, SQ*sizeof(FLOAT));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(M_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemAlloc(&tmpM_dev, SQ*sizeof(FLOAT));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(tmpM_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemAlloc(&tmpIx_dev, SQ*sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(tmpIx_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemAlloc(&tmpIy_dev, SQ*sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(tmpIy_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  /* upload datas to GPU */
  // valsを転送しなきゃだけど、valsのサイズって幾つよ？
  // →dims[0]*dims[1]*sizeof(FLOAT)=SQ*sizeof(FLOAT) だよ
  res = cuMemcpyHtoD(vals_dev, vals, SQ*sizeof(FLOAT));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD(vals) faild : res = %s\n", conv(res));
    exit(1);
  }

  void* kernel_args_x[] = {
    &vals_dev, 
    &tmpM_dev, 
    &tmpIy_dev, 
    (void*)&dims[0], 
    (void*)&ay, 
    (void*)&by,
    (void*)&dims[0],
    (void*)&dims[1]
  };

  int sharedMemBytes = 0;

  /* define CUDA block shape */
  int max_threads_num = 0;
  res = cuDeviceGetAttribute(&max_threads_num, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
  if(res != CUDA_SUCCESS){
    printf("\ncuDeviceGetAttribute() failed: res = %s\n", conv(res));
    exit(1);
  }
  
  thread_num = (dims[1] < max_threads_num) ? dims[1] : max_threads_num;
  
  block_num = dims[1] / thread_num;
  if(dims[1] % thread_num != 0) block_num++;



  res = cuLaunchKernel(
                       func_dt1d_x,      // call function
                       block_num,      // gridDimX
                       1,              // gridDimY
                       1,              // gridDimZ
                       thread_num,     // blockDimX
                       1,              // blockDimY
                       1,              // blockDimZ
                       sharedMemBytes, // sharedMemBytes
                       NULL,           // hStream
                       kernel_args_x,    // kernelParams
                       NULL            // extra
                       );
  if(res != CUDA_SUCCESS) { 
    printf("cuLaunchKernel(dt1d_x failed : res = %s\n", conv(res));
    exit(1);
  }

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(dt1d_x) failed: res = %s\n", conv(res));
    exit(1);
  }



  void* kernel_args_y[] = {
    &tmpM_dev, 
    &M_dev, 
    &tmpIx_dev, 
    (void*)&dims[0], 
    (void*)&dims[1],
    (void*)&ax, 
    (void*)&bx,
    (void*)&dims[0],
    (void*)&dims[1]
  };


  thread_num = (dims[0] < max_threads_num) ? dims[0] : max_threads_num;
  
  block_num = dims[0] / thread_num;
  if(dims[0] % thread_num != 0) block_num++;

  res = cuLaunchKernel(
               func_dt1d_y,       // call functions
               block_num,       // gridDimX
               1,               // gridDimY
               1,               // gridDimZ
               thread_num,      // blockDimX
               1,               // blockDimY
               1,               // blockDimZ
               sharedMemBytes,  // sharedMemBytes
               NULL,            // hStream
               kernel_args_y,     // kernelParams
               NULL             // extra
                       );
  if(res != CUDA_SUCCESS) { 
    printf("cuLaunchKernel(dt1d_y failed : res = %s\n", conv(res));
    exit(1);
  }


  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(dt1d_y) failed: res = %s\n", conv(res));
    exit(1);
  }


  /*************************************************************/
  /*************************************************************/
  /* original source */
  // for (int x = 0; x < dims[1]; x++)
  //   {
  //     dt1d(vals+XD, tmpM+XD, tmpIy+XD, 1, dims[0], ay, by);
  //     XD+=dims[0];
  //   }
  // for (int y = 0; y < dims[0]; y++)
  //   {
  //     dt1d(tmpM+y, M+y, tmpIx+y, dims[0], dims[1], ax, bx);
  //   }
  /*************************************************************/
  /*************************************************************/
  


  /* downloads datas from GPU */
  res = cuMemcpyDtoH(M, M_dev, SQ*sizeof(FLOAT));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH(M) failed: res = %s\n", conv(res));
    exit(1);
  } 

  res = cuMemcpyDtoH(tmpIx, tmpIx_dev, SQ*sizeof(int));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH(tmpIx) failed: res = %s\n", conv(res));
    exit(1);
  } 

  res = cuMemcpyDtoH(tmpIy, tmpIy_dev, SQ*sizeof(int));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH(tmpIy) failed: res = %s\n", conv(res));
    exit(1);
  } 


  int *IX_P = Ix;
  int *IY_P = Iy;
  int *tmpIx_P=tmpIx;
  for (int x = 0; x < dims[1]; x++) 
    {
      for (int y = 0; y < dims[0]; y++) 
        {
          *(IX_P++) = *tmpIx_P;
          *(IY_P++) = tmpIy[*tmpIx_P*dims[0]+y];
          *tmpIx_P++;
        }
    }

  /* free GPU memory */
  res = cuMemFree(M_dev);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFree(M_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(tmpM_dev);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFree(tmpM_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(tmpIx_dev);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFree(tmpIx_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(tmpIy_dev);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFree(tmpIy_dev) failed: res = %s\n", conv(res));
    exit(1);
  }


  s_free(tmpM);
  s_free(tmpIx);
  s_free(tmpIy);
  return(M);
}
