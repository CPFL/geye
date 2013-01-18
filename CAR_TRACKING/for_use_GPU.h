#ifndef _CUDA_H
#define _CUDA_H
#include <cuda.h>
#endif
#include "switch_float.h"

#ifdef __cplusplus
extern "C" {
#endif
    
struct thread_data {
    FLOAT *A;
    FLOAT *B;
    FLOAT *C;
    FLOAT *F;
    FLOAT *T;
    int A_dims[3];
    int B_dims[3];
    int C_dims[2];
};
    
    
    
    
/* define variables for using GPU */

extern CUdevice dev, dev2;
extern CUcontext ctx, ctx2;
extern CUfunction func_process_root, func_process_part;
extern CUmodule module;
extern int NR_MAXTHREADS_X, NR_MAXTHREADS_Y;

    
/* functions for using GPU and to calculate on GPU */
extern void init_cuda(void);

extern void clean_cuda(void);
    
/* function to convert CUDA error to string */
extern char *conv(unsigned int res);

/* function for GPU execution correspond to fconvsMT */
extern FLOAT ***fconvsMT_GPU(CUdeviceptr featp2_dev, FLOAT **filter,int *sym_info,int start,int end,int *A_SIZE, CUdeviceptr A_SIZE_dev, int **B_SIZE,int **M_size_array, int L_MAX, int interval, int *FSIZE, int padx, int pady, int max_X, int max_Y, int calc_flag);
    
/* definition of calc_flag */ 
#define ROOT 0
#define PART 1


/* switch define sentence  which use original source or GPU function */
//#define ORIGINAL

//#define SEPARETE_MEM


    
#ifdef __cplusplus
}
#endif

