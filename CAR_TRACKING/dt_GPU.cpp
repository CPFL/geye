#include <stdio.h>
#include <stdlib.h>

#include "for_use_GPU.h"

#define s_free(a) {free(a);a=NULL;}

FLOAT ****dt_GPU(
  int ****Ix_array,
  int ****Iy_array,
  int ***PIDX_array,
  int **size_array,
  int NoP,
  const int *numpart,
  int NoC,
  int interval,
  int L_MAX,
  int *FSIZE,
  int padx,
  int pady,
  int max_X,
  int max_Y
               )
{
  CUresult res;
  CUdeviceptr M_dev, tmpM_dev, tmpIx_dev, tmpIy_dev;

  int thread_num_x=0, thread_num_y=0;
  int block_num_x=0, block_num_y=0;

  int max_dim0 = 0, max_dim1 = 0;

  struct timeval tv;
  
  /* prepare for parallel execution */
  int sum_size_SQ = 0;
  int sum_numpart = 0;

  for(int level=interval; level<L_MAX; level++) {
    int L = level - interval;
    /**************************************************************************/
    /* loop condition */
    if( (FSIZE[level*2]+2*pady < max_Y) || (FSIZE[level*2+1]+2*padx < max_X) )
      {
        continue;
      }
    /* loop conditon */
    /**************************************************************************/

    for(int jj=0; jj<NoC; jj++) {
      for(int kk=0; kk<numpart[jj]; kk++) {
        int PIDX = PIDX_array[L][jj][kk];
        int dims0 = size_array[L][PIDX*2];
        int dims1 = size_array[L][PIDX*2+1];
        
        sum_size_SQ += dims0*dims1;
        
        /* search max values */
        max_dim0 = (max_dim0 < dims0) ? dims0 : max_dim0;
        max_dim1 = (max_dim1 < dims1) ? dims1 : max_dim1;
        
      } 
      sum_numpart += numpart[jj];
    }
  }
  
  /* allocate region each array in a lump */
  FLOAT ****M_array = (FLOAT ****)malloc((L_MAX-interval)*sizeof(FLOAT***));
  FLOAT ***sub_sub_dst_M = (FLOAT ***)malloc(NoC*(L_MAX-interval)*sizeof(FLOAT**));
  FLOAT **sub_dst_M = (FLOAT **)malloc(sum_numpart*sizeof(FLOAT*));
  FLOAT *dst_M =  (FLOAT *)malloc(sum_size_SQ*sizeof(FLOAT));

  
  FLOAT ****tmpM_array = (FLOAT ****)malloc((L_MAX-interval)*sizeof(FLOAT***));
  FLOAT ***sub_sub_dst_tmpM = (FLOAT ***)malloc(NoC*(L_MAX-interval)*sizeof(FLOAT**));
  FLOAT **sub_dst_tmpM = (FLOAT **)malloc(sum_numpart*sizeof(FLOAT*));
  FLOAT *dst_tmpM = (FLOAT *)malloc(sum_size_SQ*sizeof(FLOAT));


  int ****tmpIx_array = (int ****)malloc((L_MAX-interval)*sizeof(int***));
  int ***sub_sub_dst_tmpIx = (int ***)malloc(NoC*(L_MAX-interval)*sizeof(int**));
  int **sub_dst_tmpIx = (int **)malloc(sum_numpart*sizeof(int*));
  int *dst_tmpIx = (int *)malloc(sum_size_SQ*sizeof(int));

  
  int ****tmpIy_array = (int ****)malloc((L_MAX-interval)*sizeof(int***));
  int ***sub_sub_dst_tmpIy = (int ***)malloc(NoC*(L_MAX-interval)*sizeof(int**));
  int **sub_dst_tmpIy = (int **)malloc(sum_numpart*sizeof(int*));
  int *dst_tmpIy = (int *)malloc(sum_size_SQ*sizeof(int));

  
  /* distribute allocated region */
  unsigned long long int pointer_M = (unsigned long long int)sub_sub_dst_M;
  unsigned long long int pointer_tmpM = (unsigned long long int)sub_sub_dst_tmpM;
  unsigned long long int pointer_tmpIx = (unsigned long long int)sub_sub_dst_tmpIx;
  unsigned long long int pointer_tmpIy = (unsigned long long int)sub_sub_dst_tmpIy;
  for(int level=interval; level<L_MAX; level++) {
    int L = level - interval;
    /**************************************************************************/
    /* loop condition */
    if( (FSIZE[level*2]+2*pady < max_Y) || (FSIZE[level*2+1]+2*padx < max_X) )
      {
        continue;
      }
    /* loop conditon */
    /**************************************************************************/
    M_array[L] = (FLOAT ***)pointer_M;
    pointer_M += (unsigned long long int)(NoC*sizeof(FLOAT**));

    tmpM_array[L] = (FLOAT ***)pointer_tmpM;
    pointer_tmpM += (unsigned long long int)(NoC*sizeof(FLOAT**));

    tmpIx_array[L] = (int ***)pointer_tmpIx;
    pointer_tmpIx += (unsigned long long int)(NoC*sizeof(int**));

    tmpIy_array[L] = (int ***)pointer_tmpIy;
    pointer_tmpIy += (unsigned long long int)(NoC*sizeof(int**));
  }





  pointer_M = (unsigned long long int)sub_dst_M;
  pointer_tmpM = (unsigned long long int)sub_dst_tmpM;
  pointer_tmpIx = (unsigned long long int)sub_dst_tmpIx;
  pointer_tmpIy = (unsigned long long int)sub_dst_tmpIy;
  for(int level=interval; level<L_MAX; level++) {
    int L = level - interval;
    /**************************************************************************/
    /* loop condition */
    if( (FSIZE[level*2]+2*pady < max_Y) || (FSIZE[level*2+1]+2*padx < max_X) )
      {
        continue;
      }
    /* loop conditon */
    /**************************************************************************/

    for(int jj=0; jj<NoC; jj++) {
      int numpart_jj = numpart[jj];
      
      M_array[L][jj] = (FLOAT **)pointer_M;
      pointer_M += (unsigned long long int)(numpart_jj*sizeof(FLOAT*));
      
      tmpM_array[L][jj] = (FLOAT **)pointer_tmpM;
      pointer_tmpM += (unsigned long long int)(numpart_jj*sizeof(FLOAT*));
      
      tmpIx_array[L][jj] = (int **)pointer_tmpIx;
      pointer_tmpIx += (unsigned long long int)(numpart_jj*sizeof(int*));

      tmpIy_array[L][jj] = (int **)pointer_tmpIy;
      pointer_tmpIy += (unsigned long long int)(numpart_jj*sizeof(int*));
    }
  }
  

  pointer_M = (unsigned long long int)dst_M;
  pointer_tmpM = (unsigned long long int)dst_tmpM;
  pointer_tmpIx = (unsigned long long int)dst_tmpIx;
  pointer_tmpIy = (unsigned long long int)dst_tmpIy;
  for(int level=interval; level<L_MAX; level++) {
    int L = level - interval;
    /**************************************************************************/
    /* loop condition */
    if( (FSIZE[level*2]+2*pady < max_Y) || (FSIZE[level*2+1]+2*padx < max_X) )
      {
        continue;
      }
    /* loop conditon */
    /**************************************************************************/
    
    
    for(int jj=0; jj<NoC; jj++) {
      for(int kk=0; kk<numpart[jj]; kk++) {
        
        int PIDX = PIDX_array[L][jj][kk];
        int dims0 = size_array[L][PIDX*2];
        int dims1 = size_array[L][PIDX*2+1];
        
        M_array[L][jj][kk] = (FLOAT *)pointer_M;
        pointer_M += (unsigned long long int)(dims0*dims1*sizeof(FLOAT));
        
        tmpM_array[L][jj][kk] = (FLOAT *)pointer_tmpM;
        pointer_tmpM += (unsigned long long int)(dims0*dims1*sizeof(FLOAT));
        
        tmpIx_array[L][jj][kk] = (int *)pointer_tmpIx;
        pointer_tmpIx += (unsigned long long int)(dims0*dims1*sizeof(int));
        
        tmpIy_array[L][jj][kk] = (int *)pointer_tmpIy;
        pointer_tmpIy += (unsigned long long int)(dims0*dims1*sizeof(int));
        
      }
    }
  }
    
  /* allocate GPU memory */
  res = cuMemAlloc(&M_dev, sum_size_SQ*sizeof(FLOAT));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(M_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemAlloc(&tmpM_dev, sum_size_SQ*sizeof(FLOAT));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(tmpM_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemAlloc(&tmpIx_dev, sum_size_SQ*sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(tmpIx_dev) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuMemAlloc(&tmpIy_dev, sum_size_SQ*sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(tmpIy_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  

  int sharedMemBytes = 0;

  /* get max thread num per block */
  int max_threads_num = 0;
  res = cuDeviceGetAttribute(&max_threads_num, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev[0]);
  if(res != CUDA_SUCCESS){
    printf("\ncuDeviceGetAttribute() failed: res = %s\n", conv(res));
    exit(1);
  }
  
  
  /* prepare for launch inverse_Q */
  void* kernel_args_inverse[] = {
    &part_C_dev,
    &pm_size_array_dev,
    &part_error_array_dev,
    &part_error_array_num,
    (void*)&NoP,
    &PIDX_array_dev,
    &numpart_dev,
    (void*)&NoC,
    (void*)&max_numpart,
    (void*)&interval,
    (void*)&L_MAX
  };
  
  /* define CUDA block shape */
  int upper_limit_th_num_x = max_threads_num/(max_numpart*NoC);
  int upper_limit_th_num_y = max_threads_num/upper_limit_th_num_x;
  if(upper_limit_th_num_x < 1) upper_limit_th_num_x++;
  if(upper_limit_th_num_y < 1) upper_limit_th_num_y++;
  
  thread_num_x = (max_dim0*max_dim1 < upper_limit_th_num_x) ? (max_dim0*max_dim1) : upper_limit_th_num_x;
  thread_num_y = (max_numpart < upper_limit_th_num_y) ? max_numpart : upper_limit_th_num_y;

  block_num_x = (max_dim0*max_dim1) / thread_num_x;
  block_num_y = (max_numpart) / thread_num_y;
  if((max_dim0*max_dim1) % thread_num_x != 0) block_num_x++;
  if(max_numpart % thread_num_y != 0) block_num_y++;

  /* launch iverse_Q */
  gettimeofday(&tv_kernel_start, NULL);
  res = cuLaunchKernel(
                       func_inverse_Q[0],      // call function
                       block_num_x,         // gridDimX
                       block_num_y,         // gridDimY
                       L_MAX-interval,      // gridDimZ
                       thread_num_x,        // blockDimX
                       thread_num_y,        // blockDimY
                       NoC,                 // blockDimZ
                       sharedMemBytes,      // sharedMemBytes
                       NULL,                // hStream
                       kernel_args_inverse, // kernelParams
                       NULL                 // extra
                       );
  if(res != CUDA_SUCCESS) { 
    printf("block_num_x %d, block_num_y %d, thread_num_x %d, thread_num_y %d\n", block_num_x, block_num_y, thread_num_x, thread_num_y);
    printf("cuLaunchKernel(inverse_Q) failed : res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(inverse_Q) failed: res = %s\n", conv(res));
    exit(1);
  }
  gettimeofday(&tv_kernel_end, NULL);
  tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
  time_kernel += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;


  /* prepare for launch dt1d_x */
  void* kernel_args_x[] = {
    &part_C_dev,                  // FLOAT *src_start    
    &tmpM_dev,                    // FLOTA *dst
    &tmpIy_dev,                   // int *ptr
    &DID_4_array_dev,             // int *DID_4_array,
    &def_array_dev,               // FLOAT *def_array,
    &pm_size_array_dev,           // int *size_array     
    (void*)&NoP,                  // int NoP
    &PIDX_array_dev,              // int *PIDX_array
    &part_error_array_dev,        // int *error_array
    (void*)&part_error_array_num, // int error_array_num
    &numpart_dev,                 // int *numpart
    (void*)&NoC,                  // int NoC
    (void*)&max_numpart,          // int max_numpart
    (void*)&interval,             // int interval
    (void*)&L_MAX                 // int L_MAX
  };
  
  
  max_threads_num = 64/NoC;
  if(max_threads_num < 1) max_threads_num++;
  
  thread_num_x = (max_dim1 < max_threads_num) ? max_dim1 : max_threads_num;
  thread_num_y = (max_numpart < max_threads_num) ? max_numpart : max_threads_num;
  
  block_num_x = max_dim1 / thread_num_x;
  block_num_y = max_numpart / thread_num_y;
  if(max_dim1 % thread_num_x != 0) block_num_x++;
  if(max_numpart % thread_num_y != 0) block_num_y++;

  /* launch dt1d_x */
  gettimeofday(&tv_kernel_start, NULL);
  res = cuLaunchKernel(
                       func_dt1d_x[0],    // call function
                       block_num_x,    // gridDimX
                       block_num_y,    // gridDimY
                       L_MAX-interval, // gridDimZ
                       thread_num_x,   // blockDimX
                       thread_num_y,   // blockDimY
                       NoC,            // blockDimZ
                       sharedMemBytes, // sharedMemBytes
                       NULL,           // hStream
                       kernel_args_x,  // kernelParams
                       NULL            // extra
                       );
  if(res != CUDA_SUCCESS) { 

    printf("block_num_x %d, block_num_y %d, thread_num_x %d, thread_num_y %d\n", block_num_x, block_num_y, thread_num_x, thread_num_y);

    printf("cuLaunchKernel(dt1d_x) failed : res = %s\n", conv(res));
    exit(1);
  }
  
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(dt1d_x) failed: res = %s\n", conv(res));
    exit(1);
  }
  gettimeofday(&tv_kernel_end, NULL);
  tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
  time_kernel += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;

  
  /* prepare for launch dt1d_y */
  void* kernel_args_y[] = {
    &tmpM_dev,                    // FLOAT *src_start
    &M_dev,                       // FLOAT *dst_start
    &tmpIx_dev,                   // int *ptr_start
    &DID_4_array_dev,             // int *DID_4_array,
    &def_array_dev,               // FLOAT *def_array,
    (void*)&NoP,                  // int NoP
    &pm_size_array_dev,           // int *size_array
    &numpart_dev,                 // int *numpart,
    &PIDX_array_dev,              // int *PIDX_array,
    (void*)&NoC,                  // int NoC
    (void*)&max_numpart,          // int max_numpart
    (void*)&interval,             // int interval
    (void*)&L_MAX,                // int L_MAX
    &part_error_array_dev,        // int *error_array
    (void*)&part_error_array_num, // int error_array_num
  };
  
  
  thread_num_x = (max_dim0 < max_threads_num) ? max_dim0 : max_threads_num;
  thread_num_y = (max_numpart < max_threads_num) ? max_numpart : max_threads_num;
  
  block_num_x = max_dim0 / thread_num_x;
  block_num_y = max_numpart / thread_num_y;
  if(max_dim0 % thread_num_x != 0) block_num_x++;
  if(max_numpart % thread_num_y != 0) block_num_y++;
  

  /* prepare for launch dt1d_y */
  gettimeofday(&tv_kernel_start, NULL);
  res = cuLaunchKernel(
                       func_dt1d_y[0],    // call functions
                       block_num_x,    // gridDimX
                       block_num_y,    // gridDimY
                       L_MAX-interval, // gridDimZ
                       thread_num_x,   // blockDimX
                       thread_num_y,   // blockDimY
                       NoC,            // blockDimZ
                       sharedMemBytes, // sharedMemBytes
                       NULL,           // hStream
                       kernel_args_y,  // kernelParams
                       NULL            // extra
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
  gettimeofday(&tv_kernel_end, NULL);
  tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
  time_kernel += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;

  
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
  gettimeofday(&tv_memcpy_start, NULL);
  res = cuMemcpyDtoH(dst_M, M_dev, sum_size_SQ*sizeof(FLOAT));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH(M) failed: res = %s\n", conv(res));
    exit(1);
  } 
  
  res = cuMemcpyDtoH(dst_tmpIx, tmpIx_dev, sum_size_SQ*sizeof(int));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH(tmpIx) failed: res = %s\n", conv(res));
    exit(1);
  } 
  
  res = cuMemcpyDtoH(dst_tmpIy, tmpIy_dev, sum_size_SQ*sizeof(int));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyDtoH(tmpIy) failed: res = %s\n", conv(res));
    exit(1);
  } 
  gettimeofday(&tv_memcpy_end, NULL);
  tvsub(&tv_memcpy_end, &tv_memcpy_start, &tv);
  time_memcpy += tv.tv_sec * 1000.0 + (float)tv.tv_usec / 1000.0;


  for(int level=interval; level<L_MAX; level++) {
    int L = level - interval;
    /**************************************************************************/
    /* loop condition */
    if( (FSIZE[level*2]+2*pady < max_Y) || (FSIZE[level*2+1]+2*padx < max_X) )
      {
        continue;
      }
    /* loop conditon */
    /**************************************************************************/
    
    
    for(int jj=0; jj<NoC; jj++) {
      
      for(int kk=0; kk<numpart[jj]; kk++) {
        
        int *IX_P = Ix_array[L][jj][kk];
        int *IY_P = Iy_array[L][jj][kk];
        int *tmpIx_P=tmpIx_array[L][jj][kk];
        
        int PIDX = PIDX_array[L][jj][kk];
        int dims0 = size_array[L][PIDX*2];
        int dims1 = size_array[L][PIDX*2+1];
        
        for (int x = 0; x < dims1; x++) 
          {
            for (int y = 0; y < dims0; y++) 
              {
                *(IX_P++) = *tmpIx_P;
                *(IY_P++) = tmpIy_array[L][jj][kk][(*tmpIx_P)*(dims0)+y];
                *tmpIx_P++;       // increment address
              }
          }
      }
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
  
  
  /* free CPU memory */
  s_free(dst_tmpM);
  s_free(sub_dst_tmpM);
  s_free(sub_sub_dst_tmpM);
  s_free(tmpM_array);
  
  s_free(dst_tmpIx);
  s_free(sub_dst_tmpIx);
  s_free(sub_sub_dst_tmpIx);
  s_free(tmpIx_array);
  
  s_free(dst_tmpIy);
  s_free(sub_dst_tmpIy);
  s_free(sub_sub_dst_tmpIy);
  s_free(tmpIy_array);
  
  
  return(M_array);
}
