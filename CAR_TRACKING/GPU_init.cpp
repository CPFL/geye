#include <stdio.h>
#include <math.h>
#include "for_use_GPU.h"

/*** for debug(windows) ***//*
#include <windows.h>
#include <stdlib.h>
#include <tchar.h>*/

/*** for debug(Linux) ***/
#include <unistd.h>

/* declaration of global variables */


//extern CUdevice dev;
CUdevice dev, dev2;
CUcontext ctx, ctx2;
CUfunction func_process_root, func_process_part;
CUmodule module;
int NR_MAXTHREADS_X, NR_MAXTHREADS_Y;


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
    const char file_name[43] = "./gccDebug/GPU_function.cubin";
    
    /* initnialize GPU */
    res = cuInit(0);
    if(res != CUDA_SUCCESS){
      printf("\ncuInit failed: res = %s\n", conv(res));
      exit(1);
    }
    
    res = cuDeviceGet(&dev, 0);
    if(res != CUDA_SUCCESS){
      printf("\ncuDeviceGet(dev) failed: res = %s\n", conv(res));
      exit(1);
    }


    res = cuCtxCreate(&ctx, 0, dev);
    if(res != CUDA_SUCCESS){
      printf("\ncuCtxCreate failed: res = %s\n", conv(res));
      exit(1);
      }

    
    /* load .cubin file */
    res = cuModuleLoad(&module, file_name);
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
    
    res = cuModuleGetFunction(&func_process_root, module, "process_root");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction failed: res = %s\n", conv(res));
      exit(1);
    }

    res = cuModuleGetFunction(&func_process_part, module, "process_part");
    if(res != CUDA_SUCCESS){
      printf("\ncuGetFunction failed: res = %s\n", conv(res));
      exit(1);
    }
    
    
    
    /* get max thread num per block */
    int max_threads_num = 0;
    res = cuDeviceGetAttribute(&max_threads_num, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    if(res != CUDA_SUCCESS){
      printf("\ncuDeviceGetAttribute() failed: res = %s\n", conv(res));
      exit(1);
    }
    
    NR_MAXTHREADS_X = (int)sqrt((double)max_threads_num);
    NR_MAXTHREADS_Y = (int)sqrt((double)max_threads_num);


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

    res = cuModuleUnload(module);
    if(res != CUDA_SUCCESS){
        printf("\ncuModuleUnload failed: res = %s\n", conv(res));
        exit(1);
    }

    res = cuCtxDestroy(ctx);
    if(res != CUDA_SUCCESS){
        printf("\ncuCtxDestroy failed: res = %s\n", conv(res));
        exit(1);
        }

}/* clean_cuda */
