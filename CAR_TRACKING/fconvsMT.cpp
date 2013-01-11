//C++ library (thread-functions are only supported by windows)
#include <stdio.h>		
#include <stdlib.h>
//#include <windows.h>
//#include <process.h>
#include <math.h>

//Original header
#include "MODEL_info.h"		//File information
#include "Common.h"

#include "for_use_GPU.h"


double ***fconvsMT_GPU(CUdeviceptr featp2_dev, double **filter,int *sym_info,int start,int end,int *A_SIZE, CUdeviceptr A_SIZE_dev, int **B_SIZE,int **M_size_array, int L_MAX, int interval, int *FSIZE, int padx, int pady, int max_X, int max_Y, int calc_flag)
{
  start=start-1;
  end=end-1;
  
  const int len=end-start+1;
  double ***Output = (double ***)malloc(L_MAX*sizeof(double **));  // make double* Output[L_MAX][len]
  


  thread_data **td = (thread_data **)malloc(L_MAX*sizeof(thread_data *));  // make thread_data td[L_MAX][len] 
  thread_data *dst_td = (thread_data *)calloc(L_MAX*len, sizeof(thread_data));
  unsigned long long int ptr_td = (unsigned long long int)dst_td;
  for(int i=0; i<L_MAX; i++) {
    td[i] = (thread_data *)ptr_td;
    ptr_td += (unsigned long long int)(len*sizeof(thread_data));
  }
  
  
  int max_height=0, max_width=0;  
  
  CUresult res;
  
  int thread_num_x, thread_num_y, block_num_x, block_num_y;
  CUdeviceptr B_dims_dev;
  
  int *B_dimension = (int*)malloc(3*len*sizeof(int));
  
  CUdeviceptr B_dev;
  CUdeviceptr C_dev;  
  
  size_t SUM_SIZE_B = 0;
  size_t SUM_SIZE_C = 0;
  
  /* array in order to apply loop condition to kernel */
  int error_array_num = 0;  
  int *error_array;
  CUdeviceptr error_array_dev;
  
  /**********************************************************************/
  /* prepare output region */
  
  /* allocate output region in lump */
  double **dst_output;
  dst_output = (double **)malloc(L_MAX*len*sizeof(double *));
  if(dst_output == NULL) {
    printf("allocate dst_output failed\n");
    exit(1);
  }

  memset(dst_output, 0, L_MAX*len*sizeof(double *));  // zero clear

  /* distribution to Output[L_MAX - interval]*/
  unsigned long long int ptr_output = (unsigned long long int)dst_output;
  for(int i=0; i<L_MAX; i++) {
    Output[i] = (double **)ptr_output;
    ptr_output += (unsigned long long int)(len*sizeof(double *));
  }
  
  /* prepare output region */
  /**********************************************************************/
  
  
  
  /* prepare for launch kernel */
  for(int ii=0;ii<len;ii++)  // filter's loop(B's loop) 
	{
      /* store B dimendion in B_dimension */
      B_dimension[ii*3] = B_SIZE[ii][0];
      B_dimension[ii*3 + 1] = B_SIZE[ii][1];
      B_dimension[ii*3 + 2] = 31;
      
      
      SUM_SIZE_B += B_dimension[ii*3]*B_dimension[ii*3 + 1]*B_dimension[ii*3 + 2]*sizeof(double);
      
	}  //for(len)
  
  
  for(int level=interval; level<L_MAX; level++) {

    int L = level - interval;
    /**************************************************************************/
    /* loop conditon */
    //int level = ii + interval;

    if( (FSIZE[level*2]+2*pady < max_Y) || (FSIZE[level*2+1]+2*padx < max_X) ){
      error_array_num++;
      continue;
    }
    /* loop conditon */
    /**************************************************************************/

    for(int jj=0; jj<len; jj++) {
      
      /* compute size of output */
      
      int height, width;
      switch(calc_flag) {
      case ROOT:
        height = A_SIZE[level*3] - B_SIZE[jj][0] + 1;
        width = A_SIZE[level*3+1] - B_SIZE[jj][1] + 1;
        break;
      case PART:
        height = A_SIZE[L*3] - B_SIZE[jj][0] + 1;
        width = A_SIZE[L*3+1] - B_SIZE[jj][1] + 1;
        break;
      default:
        printf("NOT DEFINED value: calc_flag = %d\n", calc_flag);
        exit(1);
        break;
      }



      
      /* search max height and max width */
      max_height = (max_height < height) ? height : max_height;
      max_width = (max_width < width) ? width : max_width;
      
      
      if (height < 1 || width < 1)
		{
          printf("Invalid input: B should be smaller than A\n");
          printf("height %d, width %d\n", height, width);  
          exit(0);
		}
      

      switch(calc_flag){
      case ROOT:
        td[level][jj].C_dims[0]=height; 
        td[level][jj].C_dims[1]=width;
        
        SUM_SIZE_C += td[level][jj].C_dims[0]*td[level][jj].C_dims[1]*sizeof(double);
        
        M_size_array[level][jj*2]=height;
        M_size_array[level][jj*2+1]=width;
        break;
        
      case PART:      
        td[L][jj].C_dims[0]=height; 
        td[L][jj].C_dims[1]=width;
        
        SUM_SIZE_C += td[L][jj].C_dims[0]*td[L][jj].C_dims[1]*sizeof(double);
        
        M_size_array[L][jj*2]=height;
        M_size_array[L][jj*2+1]=width;
        break;
        
      default:
        printf("NOT DEFINED value: calc_flag = %d\n", calc_flag);
        exit(1);
        break;
      }

      
    }
  }
  
  
  /* save loop condition */
  res = cuMemHostAlloc((void **)&error_array, error_array_num*sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP);
  if(res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc(error_array) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  int hh=0;

  for(int level=interval; level<L_MAX; level++) {
    int L = level - interval;

    if( (FSIZE[level*2]+2*pady < max_Y) || (FSIZE[level*2+1]+2*padx < max_X) ){ /* if this evaluation formula is TRUE, the level will not be calculated */
      
      switch(calc_flag){
        
      case ROOT:
        error_array[hh] = level;
        break;
      case PART:
        
        error_array[hh] = L;
        break;
        
      default:
        printf("NOT DEFINED value: calc_flag = %d\n", calc_flag);
        exit(1);
        break;
      }
      
      hh++;
      if(hh > error_array_num) {
        printf("beyond error_array_num!\n");
        exit(1);
      }
    }
  }
  

  /* define CUDA block shape */
  int max_threads_num = 0;
  res = cuDeviceGetAttribute(&max_threads_num, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
  if(res != CUDA_SUCCESS){
    printf("\ncuDeviceGetAttribute() failed: res = %s\n", conv(res));
    exit(1);
  }


  /* calculate max size of each block dimension */
  NR_MAXTHREADS_X = (int)sqrt((double)max_threads_num/len);
  NR_MAXTHREADS_Y = (int)sqrt((double)max_threads_num/len);


  thread_num_x = (max_width < NR_MAXTHREADS_X) ? max_width : NR_MAXTHREADS_X;
  thread_num_y = (max_height < NR_MAXTHREADS_Y) ? max_height : NR_MAXTHREADS_Y;
  
  block_num_x = max_width / thread_num_x;
  block_num_y = max_height / thread_num_y;
  if(max_width % thread_num_x != 0) block_num_x++;
  if(max_height % thread_num_y != 0) block_num_y++;
  
  /* allocate GPU memory */

  res = cuMemAlloc(&B_dev, SUM_SIZE_B);
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(B_dev) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuMemAlloc(&B_dims_dev, 3*len*sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(B_dims) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuMemAlloc(&error_array_dev, error_array_num*sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(error_array_dev) failed: res = %s\n", conv(res));
    exit(1);
  }


  /* upload data to GPU memory */

  /* upload filter */
  res = cuMemcpyHtoD(B_dev, filter[start], SUM_SIZE_B);
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(B_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  /* upload error_array */
  res = cuMemcpyHtoD(error_array_dev, error_array, error_array_num*sizeof(int));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD(error_array_dev) failed: res = %s\n", conv(res));
    exit(1);
  }


  /* allocate output region on CPU memory */
  double *dst_C;
  res = cuMemHostAlloc((void **)&dst_C, SUM_SIZE_C, CU_MEMHOSTALLOC_DEVICEMAP);
  if(res != CUDA_SUCCESS){
    printf("cuMemHostAlloc(dst_C) failed: res = %s\n", conv(res));
    exit(1);
  }

  memset(dst_C, 0, SUM_SIZE_C); //zero clear

  /* distribution */
  unsigned long long int pointer = (unsigned long long int)dst_C;
  for(int level=interval; level<L_MAX; level++) {
    int L = level - interval;
    for(int jj=0; jj<len; jj++) {

      switch(calc_flag) {
      case ROOT:
        td[level][jj].C = (double *)pointer;
        pointer += (unsigned long long int)(td[level][jj].C_dims[0]*td[level][jj].C_dims[1]*sizeof(double));
        break;

      case PART:
        td[L][jj].C = (double *)pointer;
        pointer += (unsigned long long int)(td[L][jj].C_dims[0]*td[L][jj].C_dims[1]*sizeof(double));
        break;

      default:
        printf("NOT DEFINED value: calc_flag = %d\n", calc_flag);
        exit(1);
        break;
        
      }

    }
  }


  /* allocate output region on GPU memory */
  res = cuMemAlloc(&C_dev, SUM_SIZE_C);
  if(res != CUDA_SUCCESS){
    printf("cuMemAlloc(C_dev) failed: res = %s\n", conv(res));
    exit(1);
  }
#if 0
  res = cuMemsetD32(C_dev, (double)0, (size_t)(SUM_SIZE_C / sizeof(double)));
  if(res != CUDA_SUCCESS){
    printf("cuMemsetD32(C_dev) failed: res = %s\n", conv(res));
    exit(1);
  }
#else
  res = cuMemcpyHtoD(C_dev, dst_C, SUM_SIZE_C);
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD(C_dev) failed: res = %s\n", conv(res));
    exit(1);
  }
#endif

  res = cuMemcpyHtoD(B_dims_dev, B_dimension, 3*len*sizeof(int));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyHtoD(B_dims) failed: res = %s\n", conv(res));
    exit(1);
  }


  /* launch kernel
     grid shape : block_num_x * block_num_y * L_MAX, 
     block shape : thread_num_x * thread_num_y * len 
  */              
  /* dealing with 1 feature(A) by 1 z_dimension of grid */
  /* dealing with 1 model(B) by 1 z_dimension of block */

  void *kernel_args[] = {
    &featp2_dev, 
    &B_dev, 
    &C_dev, 
    &A_SIZE_dev, 
    &B_dims_dev, 
    (void *)&len, 
    (void *)&interval, 
    (void *)&L_MAX, 
    &error_array_dev, 
    (void *)&error_array_num
  };
  
  int sharedMemBytes = 0;

#if 0
  printf("sizeof(&feadp2_dev) %lu\n", sizeof(&featp2_dev));
  printf("sizeof(&B_dev) %lu\n", sizeof(&B_dev));
  printf("sizeof(&C_dev) %lu\n", sizeof(&C_dev));
  printf("sizeof(&A_SIZE_dev) %lu\n", sizeof(&A_SIZE_dev));
  printf("sizeof(&B_dims_dev) %lu\n", sizeof(&B_dims_dev));
  printf("sizeof(&len) %lu\n", sizeof(&len));
  printf("sizeof(&interval) %lu\n", sizeof(&interval));
  printf("sizeof(&L_MAX) %lu\n", sizeof(&L_MAX));
  printf("sizeof(&error_array_dev) %lu\n", sizeof(&error_array_dev));
  printf("sizeof(&error_array_num) %lu\n", sizeof(&error_array_num));
  printf("sizeof(void *) %lu\n", sizeof(void *));

  printf("---------------------------------\n");
  printf("block_num_x %d\n", block_num_x);
  printf("block_num_y %d\n", block_num_y);
  printf("L_MAX %d\n", L_MAX);
  printf("thread_num_x %d\n", thread_num_x);
  printf("thread_num_y %d\n", thread_num_y);
  printf("len %d\n", len);
  printf("sharedMemBytes %d\n", sharedMemBytes);
  printf("error_array_num %d\n", error_array_num);

  printf("---------------------------------\n");
  printf("sizeof(int) %lu\n", sizeof(int));
  printf("sizeof(unsigned int) %lu\n", sizeof(unsigned int));

  printf("---------------------------------\n");
  printf("sizeof(block_num_x) %lu\n", sizeof(block_num_x));
  printf("sizeof(block_num_y) %lu\n", sizeof(block_num_y));
  printf("sizeof(L_MAX) %lu\n", sizeof(L_MAX));
  printf("sizeof(thread_num_x) %lu\n", sizeof(thread_num_x));
  printf("sizeof(thread_num_y) %lu\n", sizeof(thread_num_y));
  printf("sizeof(len) %lu\n", sizeof(len));
  printf("sizeof(sharedMemBytes) %lu\n", sizeof(sharedMemBytes));

  printf("---------------------------------\n");
  printf("sizeof(CUstream) %lu\n", sizeof(CUstream));
  printf("sizeof(void**) %lu\n", sizeof(void**));
  printf("sizeof(kernel_args) %lu\n", sizeof(kernel_args));
  printf("sizeof(NULL) %lu\n", sizeof(NULL));
#endif

#if 0
  int ctnumdb=0;
  res = cuDeviceGetAttribute(&ctnumdb, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev);
  printf("max block dim x %d\n", ctnumdb);


  res = cuDeviceGetAttribute(&ctnumdb, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev);
  printf("max block dim y %d\n", ctnumdb);

  res = cuDeviceGetAttribute(&ctnumdb, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev);
  printf("max block dim z %d\n", ctnumdb);

  printf("thread_num_x %d\n", thread_num_x);
  printf("thread_num_y %d\n",thread_num_y);
  printf("len %d\n", len);

  printf("---------------------------------\n");

  res = cuDeviceGetAttribute(&ctnumdb, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
  printf("max grid dim x %d\n", ctnumdb);

  res = cuDeviceGetAttribute(&ctnumdb, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
  printf("max grid dim y %d\n", ctnumdb);

  res = cuDeviceGetAttribute(&ctnumdb, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
  printf("max grid dim z %d\n", ctnumdb);

  printf("block_num_x %d\n", block_num_x);
  printf("block_num_y %d\n", block_num_y);
  printf("L_MAX %d\n", L_MAX);
#endif


  switch(calc_flag) {  
  case ROOT: 
    // res = cuLaunchKernel(
    //                      func_process_root, // call function
    //                      block_num_x,       // gridDimX
    //                      block_num_y,       // gridDimY
    //                      L_MAX,             // gridDimZ
    //                      thread_num_x,      // blockDimX
    //                      thread_num_y,      // blockDimY
    //                      len,               // blockDimZ
    //                      sharedMemBytes,    // sharedMemBytes
    //                      NULL,              // hStream
    //                      kernel_args,       // kernelParams
    //                      NULL               // extra
    //                      );
    res = cuLaunchKernel(
                         func_process_root, // call function
                         block_num_x,       // gridDimX
                         block_num_y,       // gridDimY
                         L_MAX*len,             // gridDimZ
                         thread_num_x,      // blockDimX
                         thread_num_y,      // blockDimY
                         1,                 // blockDimZ
                         sharedMemBytes,    // sharedMemBytes
                         NULL,              // hStream
                         kernel_args,       // kernelParams
                         NULL               // extra
                         );
    if(res != CUDA_SUCCESS){
      printf("cuLaunchKernel(root) failed: res = %s\n", conv(res));
      exit(1);
    }
    break;
  case PART: 
    // res = cuLaunchKernel(
    //                      func_process_part, // call function
    //                      block_num_x,       // gridDimX
    //                      block_num_y,       // gridDimY
    //                      L_MAX,             // gridDimZ
    //                      thread_num_x,      // blockDimX
    //                      thread_num_y,      // blockDimY
    //                      len,               // blockDimZ
    //                      sharedMemBytes,    // sharedMemBytes
    //                      NULL,              // hStream
    //                      kernel_args,       // kernelParams
    //                      NULL               // extra
    //                      );
    res = cuLaunchKernel(
                         func_process_part, // call function
                         block_num_x,       // gridDimX
                         block_num_y,       // gridDimY
                         L_MAX*len,             // gridDimZ
                         thread_num_x,      // blockDimX
                         thread_num_y,      // blockDimY
                         1,               // blockDimZ
                         sharedMemBytes,    // sharedMemBytes
                         NULL,              // hStream
                         kernel_args,       // kernelParams
                         NULL               // extra
                         );
    if(res != CUDA_SUCCESS){
      printf("cuLaunchKernel(part) failed: res = %s\n", conv(res));
      exit(1);
    }
    break;
  default:
    printf("NOT DEFINED value: calc_flag = %d\n", calc_flag);
    exit(1);
    break;
  }


  /* synchronize GPU threads */
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS){
    printf("cuCtxSynchronize(process) failed: res = %s\n", conv(res));
    exit(1);
  }


  /* download C from GPU */
  res = cuMemcpyDtoH((void *)dst_C, C_dev, SUM_SIZE_C);
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH(dst_C) failed: res = %s\n", conv(res));
    exit(1);
  }

  
  //close handle and get output 
  for(int level=interval; level<L_MAX; level++) {
    int L = level - interval;
    /**************************************************************************/
    /* loop condition */
    //    int level = ii + interval;
    if( (FSIZE[level*2]+2*pady < max_Y) || (FSIZE[level*2+1]+2*padx < max_X) )
      {
        continue;
      }
    /* loop conditon */
    /**************************************************************************/
    for(int jj=0; jj<len; jj++) {
      //       if(level == interval && jj == 0 && calc_flag == ROOT){
      //         printf("sizeof(double) %llu\n", (unsigned long long int)sizeof(double));
      //         printf("sizeof(double*) %llu\n", (unsigned long long int)sizeof(double*));
      //         printf("%f CPU \n", *td[level][jj].C);
      //         printf("%f CPU \n", dst_C[0]);
      //         //         printf("%llu CPU ad\n", (unsigned long long int)td[level][jj].C);
      //         //         printf("%llu CPU ad\n", (unsigned long long int)dst_C);
      //       }
      
      switch(calc_flag){
        
      case ROOT:
        Output[level][jj] = td[level][jj].C;
        break;

      case PART:
        Output[L][jj] = td[L][jj].C;
        break;

      default:
        printf("NOT DEFINED value: calc_flag = %d\n", calc_flag);
        exit(1);
        break;
      }
    }
  }
  


  /* free GPU memory */
  res = cuMemFree(B_dims_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(B_dims_dev) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuMemFree(B_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(B_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(C_dev);
  if(res != CUDA_SUCCESS){
    printf("cuMemFree(C_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(error_array_dev);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFree(error_array_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  /* free CPU memory */
  res = cuMemFreeHost((void *)error_array);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFreeHost(error_array) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  s_free(B_dimension);
  s_free(td[0]);	  
  s_free(td);	

  
  return(Output);
  
}
