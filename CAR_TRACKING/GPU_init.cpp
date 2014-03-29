#include <stdio.h>
#include <math.h>
#include "for_use_GPU.h"
#include "cutil.h"
#include "drvapi_error_string.h"
#include <cuda_runtime_api.h>
#include "switch_release.h"

#define SIZE_FEATP2 100000000
#define SIZE_A_SIZE 1000
#define SIZE_B 100000
#define SIZE_B_DIMS 1000
#define SIZE_ERROR_ARRAY 100
#define SIZE_C 50000000
#define SIZE_PM 10000
#define SIZE_DEF 1000
#define SIZE_NUMPART 100
#define SIZE_PIDX 10000
#define SIZE_DID 10000
#define SIZE_M 30000000
#define SIZE_TMPM 30000000
#define SIZE_TMPIX 30000000
#define SIZE_TMPIY 30000000
/*** for debug(windows) ***//*
#include <windows.h>
#include <stdlib.h>
#include <tchar.h>*/

/*** for debug(Linux) ***/
#include <unistd.h>

/* declaration of global variables */


//extern CUdevice dev;
#define conv(arg) getCudaDrvErrorString(arg)
CUdevice *dev;
CUcontext *ctx;
//CUdevice dev, dev2;
//CUcontext ctx, ctx2;
CUfunction *func_process_root, *func_process_part, *func_dt1d_x, *func_dt1d_y, *func_calc_a_score, *func_inverse_Q, *func_calc_feature;
CUmodule *module;
int *NR_MAXTHREADS_X, *NR_MAXTHREADS_Y;
CUdeviceptr *A_SIZE_dev, *featp2_dev, *B_dev, *B_dims_dev, *fconvs_error_array_dev, *fconvs_C_dev, *part_C_dev, *part_error_array_dev, *pm_size_array_dev, *PIDX_array_dev, *def_array_dev, *DID_4_array_dev, *numpart_dev,*M_dev, *tmpM_dev, *tmpIx_dev, *tmpIy_dev;
/*** for debug(windows) ***//*
#define _MAX_PATH 256
#define _MAX_DIR 256
#define _MAX_DRIVE 8 */
                  
//TCHAR szAppDir[_MAX_PATH];  // アプリケーションが起動されたディレクトリ
//TCHAR szFull[_MAX_PATH];    // 起動されたアプリケーションのフルパス名
//TCHAR szDrive[_MAX_DRIVE];  // 起動されたアプリケーションのドライブ名
//TCHAR szDir[_MAX_DIR];      // 起動されたアプリケーションのディレクトリ名
                  

/*****************************************************************/
/* init_cuda

   initialization device to use CUDA function 
*/
/*****************************************************************/
void init_cuda(void)
{


    CUresult res;
    //const char file_name[43] = "./gccDebug/GPU_function.cubin";
#ifdef RELEASE
    const char file_name[256] = "/usr/local/geye/bin/car_detecter/GPU_function.cubin";
#else
    const char file_name[43] = "./gccRelease/GPU_function.cubin";
#endif
    int i;
    /* initnialize GPU */
    res = cuInit(0);
    if(res != CUDA_SUCCESS){
      printf("\ncuInit failed: res = %s\n", conv(res));
      exit(1);
    }

  /* count the number of usable GPU */
    res = cuDeviceGetCount(&device_num);
    if(res != CUDA_SUCCESS) {
      printf("cuDeviceGetCount() failed: res = %s\n", conv(res));
      exit(1);
    }

#ifdef PRINT_INFO
    printf("%d GPUs found\n", device_num);
#endif
    //    device_num = 4;
  /* get device */
    dev = (CUdevice*)malloc(device_num*sizeof(CUdevice));
    for(int i=0; i<device_num; i++) {
    //    res = cuDeviceGet(&dev[i], 0);
      res = cuDeviceGet(&dev[i], i);
      if(res != CUDA_SUCCESS) {
      printf("cuDeviceGet(dev[%d]) failed: res = %s\n", i, conv(res));
      exit(1);
    }
  }

#if 0
  /* check whether peer-to-peer access between GPUs is possible */
  int canAccessPeer=0;
  cudaDeviceCanAccessPeer(&canAccessPeer, dev[0], dev[1]);
  if(canAccessPeer ==1 )
    printf("p2p access dev[0] -> dev[1] is ENable\n");
  else
    printf("p2p access dev[0] -> dev[1] is DISable\n"); 

  cudaDeviceCanAccessPeer(&canAccessPeer, dev[1], dev[0]);
  if(canAccessPeer ==1 )
    printf("p2p access dev[1] -> dev[0] is ENable\n");
  else
    printf("p2p access dev[1] -> dev[0] is DISable\n"); 
#endif


  ctx = (CUcontext*)malloc(device_num*sizeof(CUcontext));

  module = (CUmodule*)malloc(device_num*sizeof(CUmodule));

  func_process_root = (CUfunction*)malloc(device_num*sizeof(CUfunction));
  func_process_part = (CUfunction*)malloc(device_num*sizeof(CUfunction));
  func_dt1d_x = (CUfunction*)malloc(device_num*sizeof(CUfunction));
  func_dt1d_y = (CUfunction*)malloc(device_num*sizeof(CUfunction));
  func_calc_a_score = (CUfunction*)malloc(device_num*sizeof(CUfunction));
  func_inverse_Q = (CUfunction*)malloc(device_num*sizeof(CUfunction));
  func_calc_feature = (CUfunction*)malloc(device_num*sizeof(CUfunction));



  for(int i=0; i<device_num; i++) {

    res = cuCtxCreate(&ctx[i], 0, dev[i]);
    if(res != CUDA_SUCCESS) {
      printf("cuCtxCreate(ctx[%d]) failed: res = %s\n", i, conv(res));
      exit(1);
    }
  }


  for(int i=0; i<device_num; i++) {

    res = cuCtxSetCurrent(ctx[i]);
    if(res != CUDA_SUCCESS) {
       printf("cuCtxSetCurrent(ctx[%d]) failed: res = %s\n", i, conv(res));
       exit(1);
     }


    /* load .cubin file */
    res = cuModuleLoad(&module[i], file_name);
    if(res != CUDA_SUCCESS){
      printf("\ncuModuleLoad failed: res = %s\n", conv(res));
      /*** for debug(windows) ***//*  
                         // 起動されたアプリケーションのフルパス名を取得
                         ::GetModuleFileName(NULL, szFull, sizeof(szFull) / sizeof(TCHAR));
                         
                         // フルパス名をドライブ名やディレクトリ名部分に分解
                         _tsplitpath(szFull, szDrive, szDir, NULL, NULL);
                         
                         // ドライブ名とディレクトリ名部分を連結
                         _tmakepath(szAppDir, szDrive, szDir, NULL, NULL);
                         
                         MessageBox(NULL, szAppDir, (LPCWSTR)" ", MB_OK);*/
      /*** for debug(Linux) ***//*
           char pathname[512]="";
           // get current directory name
           getcwd(pathname, 512);
           // display current directory
           printf("current directory : %s\n", pathname);*/
      exit(1);
    }

    res = cuModuleGetFunction(&func_process_root[i], module[i], "process_root");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction(process_root) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuModuleGetFunction(&func_process_part[i], module[i], "process_part");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction(process_part) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuModuleGetFunction(&func_inverse_Q[i], module[i], "inverse_Q");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction(inverse_Q) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuModuleGetFunction(&func_dt1d_x[i], module[i], "dt1d_x");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction(dt1d_x) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuModuleGetFunction(&func_dt1d_y[i], module[i], "dt1d_y");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction(dt1d_y) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuModuleGetFunction(&func_calc_a_score[i], module[i], "calc_a_score");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction(calc_a_score) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuModuleGetFunction(&func_calc_feature[i], module[i], "calc_feature");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction(calc_feature) failed: res = %s\n", conv(res));
      exit(1);
    }

  }






  /* allocate GPU memory */

  A_SIZE_dev = (CUdeviceptr *)malloc(device_num*sizeof(CUdeviceptr));
  featp2_dev = (CUdeviceptr *)malloc(device_num*sizeof(CUdeviceptr));
  B_dev = (CUdeviceptr *)malloc(device_num*sizeof(CUdeviceptr));
  B_dims_dev = (CUdeviceptr *)malloc(device_num*sizeof(CUdeviceptr));
  fconvs_error_array_dev = (CUdeviceptr *)malloc(device_num*sizeof(CUdeviceptr));
  fconvs_C_dev = (CUdeviceptr *)malloc(device_num*sizeof(CUdeviceptr));
  part_error_array_dev = (CUdeviceptr *)malloc(sizeof(CUdeviceptr) * device_num);
  part_C_dev = (CUdeviceptr *)malloc(sizeof(CUdeviceptr) * device_num);
  pm_size_array_dev = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));
  PIDX_array_dev = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));
  def_array_dev = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));
  DID_4_array_dev = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));
  numpart_dev  = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));
  M_dev = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));
  tmpM_dev = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));
  tmpIx_dev = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));
  tmpIy_dev  = (CUdeviceptr*)malloc(device_num*sizeof(CUdeviceptr));



  for(int i=0; i<device_num; i++) {

    res = cuCtxSetCurrent(ctx[i]);
    if(res != CUDA_SUCCESS) {
       printf("cuCtxSetCurrent(ctx[%d]) failed: res = %s\n", i, conv(res));
       exit(1);
     }

    res = cuMemAlloc(&featp2_dev[i], SIZE_FEATP2);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(featp2_dev) failed: res = %s\n", conv(res));
      exit(1);
    }


    res = cuMemAlloc(&A_SIZE_dev[i], SIZE_A_SIZE);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(A_SIZE_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&B_dev[i], SIZE_B);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(B_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&B_dims_dev[i], SIZE_B_DIMS);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(B_dims_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&fconvs_error_array_dev[i], SIZE_ERROR_ARRAY);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(fconvs_error_array_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&fconvs_C_dev[i], SIZE_C);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(fconvs_C_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&part_C_dev[i], SIZE_C);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(part_C_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&part_error_array_dev[i], SIZE_ERROR_ARRAY);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(part_error_array_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&pm_size_array_dev[i], SIZE_PM);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(pm_size_array_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&def_array_dev[i], SIZE_DEF);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(def_array_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&numpart_dev[i], SIZE_NUMPART);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(numpart_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&PIDX_array_dev[i], SIZE_PIDX);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(PIDX_array_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&DID_4_array_dev[i], SIZE_DID);
    if(res != CUDA_SUCCESS) {
      printf("cuMemAlloc(DID_4__array_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&M_dev[i], SIZE_M);
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(M_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&tmpM_dev[i], SIZE_TMPM);
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(tmpM_dev) failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuMemAlloc(&tmpIx_dev[i], SIZE_TMPIX);
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(tmpIx_dev) failed: res = %s\n", conv(res));
      exit(1);
    }
  
    res = cuMemAlloc(&tmpIy_dev[i], SIZE_TMPIY);
    if(res != CUDA_SUCCESS){
      printf("cuMemAlloc(tmpIy_dev) failed: res = %s\n", conv(res));
      exit(1);
    }




  }



  NR_MAXTHREADS_X = (int*)malloc(device_num*sizeof(int));
  NR_MAXTHREADS_Y = (int*)malloc(device_num*sizeof(int));


  for(int i=0; i<device_num; i++) {
    
    /* get max thread num per block */
    int max_threads_num = 0;
    res = cuDeviceGetAttribute(&max_threads_num, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev[i]);
    if(res != CUDA_SUCCESS){
      printf("\ncuDeviceGetAttribute() failed: res = %s\n", conv(res));
      exit(1);
    }
    
    NR_MAXTHREADS_X[i] = (int)sqrt((double)max_threads_num);
    NR_MAXTHREADS_Y[i] = (int)sqrt((double)max_threads_num);

  }

    res = cuCtxSetCurrent(ctx[0]);
    if(res != CUDA_SUCCESS) {
       printf("cuCtxSetCurrent(ctx[%d]) failed: res = %s\n", i, conv(res));
       exit(1);
     }
#if 0
    /*** for debug ***/
    /* show device information */
    printf("************ device information ************\n");
    char devname[256];
    cuDeviceGetName(devname, sizeof(devname), dev);
    printf("Device Name : %s\n", devname);
    printf("--------------------------------------------\n");

    int pi;
    cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    printf("Max Threads per Block : %d\n", pi);
    printf("--------------------------------------------\n");

    cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev);
    if(pi != 0) printf("device is integrated with the host memory system\n");
    else printf("device is NOT integrated with the host memory system\n");
    printf("--------------------------------------------\n");

    cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
    if(pi != 0) printf("device can map host memory\n");
    else printf("device CANNOT map host memory\n");
    printf("--------------------------------------------\n");

    cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
    if(pi != 0) printf("device shares a unified address space\n");
    else printf("device DOES NOT share a unified address space\n");
    printf("--------------------------------------------\n");

    cuDeviceCanAccessPeer(&pi, dev2, dev);
    if(pi != 0) printf("dev are capable of directly accessing memory from dev2\n");
    else printf("dev are NOT capable of directly accessing memory from dev2\n");
    printf("--------------------------------------------\n");

    int major = 0, minor = 0;
    cuDeviceComputeCapability(&major, &minor, dev);
    printf("Compute Capability : major = %d, minor = %d\n", major, minor);
    printf("--------------------------------------------\n");

    cuDeviceGetCount(&pi);
    printf("Available device number : %d\n", pi);

    printf("********************************************\n");

    printf("if you want to exit, type 'q' then Push Enter key\n");
    char check_exit;
    check_exit = getchar();
    if(check_exit == 'q') exit(1);

#endif

}/* init_cuda */


/*****************************************************************/
/* clean_cuda

   cleaning up after using GPU
*/
/*****************************************************************/
void clean_cuda(void)
{
    CUresult res;

#if 0
    res = cuCtxPushCurrent(ctx);
    if(res != CUDA_SUCCESS){
      printf("cuCtxPushCurrent(ctx) failed: res = %s\n", conv(res));
      exit(1);
    }
#endif


  
  for(int i=0; i<device_num; i++){

    res = cuCtxSetCurrent(ctx[i]);
    if(res != CUDA_SUCCESS) {
       printf("cuCtxSetCurrent(ctx[%d]) failed: res = %s\n", i, conv(res));
       exit(1);
     }


    res = cuMemFree(featp2_dev[i]);
    if(res != CUDA_SUCCESS) {
	printf("cuMemFree(featp2_dev) failed: res = %s\n", conv(res));
	exit(1);
      }

    res = cuMemFree(A_SIZE_dev[i]);
    if(res != CUDA_SUCCESS) {
	printf("cuMemFree(A_SIZE_dev) failed: res = %s\n", conv(res));
	exit(1);
      }

    res = cuMemFree(B_dev[i]);
    if(res != CUDA_SUCCESS) {
	printf("cuMemFree(B_dev) failed: res = %s\n", conv(res));
	exit(1);
      }

    res = cuMemFree(B_dims_dev[i]);
    if(res != CUDA_SUCCESS) {
	printf("cuMemFree(B_dev) failed: res = %s\n", conv(res));
	exit(1);
      }

    res = cuMemFree(fconvs_error_array_dev[i]);
    if(res != CUDA_SUCCESS) {
	printf("cuMemFree(fconvs_error_array_dev) failed: res = %s\n", conv(res));
	exit(1);
      }

    res = cuMemFree(fconvs_C_dev[i]);
    if(res != CUDA_SUCCESS) {
	printf("cuMemFree(fconvs_C_dev) failed: res = %s\n", conv(res));
	exit(1);
      }

    res = cuMemFree(part_C_dev[i]);
    if(res != CUDA_SUCCESS) {
	printf("cuMemFree(part_C_dev) failed: res = %s\n", conv(res));
	exit(1);
      }

    res = cuMemFree(part_error_array_dev[i]);
    if(res != CUDA_SUCCESS) {
	printf("cuMemFree(part_error_array_dev) failed: res = %s\n", conv(res));
	exit(1);
      }

    res = cuMemFree(pm_size_array_dev[i]);
    if(res != CUDA_SUCCESS){
        printf("cuMemFree(pm_size_array_dev) failed: res = %s\n", conv(res));
        exit(1);
      }

     res = cuMemFree(def_array_dev[i]);
     if(res != CUDA_SUCCESS) {
       printf("cuMemFree(def_array_dev) failed: res = %s\n", conv(res));
       exit(1);
      }

      res = cuMemFree(numpart_dev[i]);
      if(res != CUDA_SUCCESS) {
        printf("cuMemFree(numpart_dev) failed: res = %s\n", conv(res));
        exit(1);
      }

      res = cuMemFree(M_dev[i]);
      if(res != CUDA_SUCCESS) {
        printf("cuMemFree(M_dev) failed: res = %s\n", conv(res));
        exit(1);
      }
  
      res = cuMemFree(tmpM_dev[i]);
      if(res != CUDA_SUCCESS) {
        printf("cuMemFree(tmpM_dev) failed: res = %s\n", conv(res));
        exit(1);
      }
  
      res = cuMemFree(tmpIx_dev[i]);
      if(res != CUDA_SUCCESS) {
        printf("cuMemFree(tmpIx_dev) failed: res = %s\n", conv(res));
        exit(1);
      }
  
      res = cuMemFree(tmpIy_dev[i]);
      if(res != CUDA_SUCCESS) {
        printf("cuMemFree(tmpIy_dev) failed: res = %s\n", conv(res));
        exit(1);
      }

   }


  for(int i=0; i<device_num; i++){
    res = cuModuleUnload(module[i]);
    if(res != CUDA_SUCCESS){
        printf("\ncuModuleUnload failed: res = %s\n", conv(res));
        exit(1);
    }
 }
  printf("module unloaded\n");

  for(int i=0; i<device_num; i++){
    res = cuCtxDestroy(ctx[i]);
    if(res != CUDA_SUCCESS){
        printf("\ncuCtxDestroy failed: res = %s\n", conv(res));
        exit(1);
    }
  }
  printf("context destroyed\n");
    free(featp2_dev);
    free(A_SIZE_dev);
    free(B_dev);
    free(B_dims_dev);
    free(fconvs_error_array_dev);
    free(fconvs_C_dev);
    free(NR_MAXTHREADS_X);
    free(NR_MAXTHREADS_Y);
    free(func_process_root);
    free(func_process_part);
    free(func_dt1d_x); 
    free(func_dt1d_y);
    free(func_calc_a_score);
    free(func_inverse_Q);
    free(func_calc_feature);
    free(part_C_dev);
    free(part_error_array_dev);
    free(pm_size_array_dev);
    free(def_array_dev);
    free(numpart_dev);
    free(DID_4_array_dev);
    free(PIDX_array_dev); 
    free(M_dev);
    free(tmpM_dev);
    free(tmpIx_dev); 
    free(tmpIy_dev); 
    free(module);
    free(dev);
    free(ctx);
    printf("clean_cuda finished\n");
}/* clean_cuda */
