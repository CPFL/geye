#include<stdio.h>
#include"for_use_GPU.h"


//thread process
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolve A and B(non_symmetric)
//unsigned __stdcall process(void *thread_arg) {

/********************************************/
/* function for calculating root */
/********************************************/
extern "C"
__global__
void
process_root(
 double *A,  
 double *B, 
 double *C, 
 int *A_dims_array, 
 int *B_dims_array, 
 int len,
 int interval, 
 int L_MAX,
 int *error_array,
 int error_array_num
) 
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  // int ii = threadIdx.z;
  // int level = blockIdx.z; 
  int ii = blockIdx.z % len;
  int level = blockIdx.z / len;

  
  int A_dims[3] = { A_dims_array[level*3], A_dims_array[level*3+1], A_dims_array[level*3+2] };
  int B_dims[3] = { B_dims_array[ii*3], B_dims_array[ii*3+1], B_dims_array[ii*3+2] };
  int C_dims[2] = { A_dims[0] - B_dims[0] + 1, A_dims[1] - B_dims[1] + 1 };
  
  
  if(0 <= ii && ii < len && 0 <= idx_x && idx_x < C_dims[1] && 0 <= idx_y && idx_y < C_dims[0] && interval <= level && level < L_MAX ) {    
    
    int num_features = A_dims[2];
    const int A_SQ = A_dims[0]*A_dims[1];
    const int B_SQ = B_dims[0]*B_dims[1];
    double add_val = 0;
    
    int x = idx_x;
    int y = idx_y;
    int XA0 = A_dims[0]*x;

    
    /* apply loop condition */
    for(int i=0; i<error_array_num; i++){
      if(error_array[i] == level){
        return;
      }
    }
    
    
    
    /* adjust the location of pointer of C */
    double *dst;
    unsigned long long int pointer = (unsigned long long int)C;

    for(int a=interval; a<level; a++){
      for(int b=0; b<len; b++){
        int height = A_dims_array[a*3] - B_dims_array[b*3] + 1; 
        int width = A_dims_array[a*3 + 1] - B_dims_array[b*3 + 1] + 1;
        
        /* error semantics */
        if(height < 1 || width < 1){
          printf("Invalid input in GPU\n");
          return;
        }
        
        pointer += (unsigned long long int)(height*width*sizeof(double));
       
      }
    }

    for(int b=0; b<ii; b++){
      int height = A_dims_array[level*3] - B_dims_array[b*3] + 1;
      int width  = A_dims_array[level*3 + 1] - B_dims_array[b*3 + 1] + 1;

      /* error semantics */
      if(height < 1 || width < 1){
        printf("Invalid input in GPU\n");
        return;
      }
      
      pointer += (unsigned long long int)(height*width*sizeof(double));
    }
    
    dst = (double *)pointer;
    
    
    //    if(ii==0 && idx_x==0 && idx_y==0 && level == interval){
    //       printf("C       %llu \n", (unsigned long long int)C);
    //       printf("dst     %llu \n", (unsigned long long int)dst);
    //       printf("pointer %llu \n", (unsigned long long int)pointer);
    //       // printf("sizeof  %llu \n", sizeof(unsigned long long int));
    //       printf("%llu %llu %llu %llu \n", (unsigned long long int)C, (unsigned long long int)dst, (unsigned long long int)pointer, (unsigned long long int)sizeof(unsigned long long int)); 
    //  }
    

    /* adjust the location of pointer of A */
    unsigned long long int pointerA = (unsigned long long int)A;
    for(int a=0; a<level; a++) {
      pointerA += (unsigned long long int)(A_dims_array[a*3]*A_dims_array[a*3 + 1]*A_dims_array[a*3 + 2]*sizeof(double));
    } 
    
    
    /* adjust the location of pointer of B */
    unsigned long long int pointerB = (unsigned long long int)B;
    for(int b=0; b<ii; b++) {
      pointerB += (unsigned long long int)(B_dims_array[b*3]*B_dims_array[b*3 + 1]*B_dims_array[b*3 + 2]*sizeof(double));
    } 

            
    for(int f = 0; f < num_features; f++) // num_features = 31
      {  
        //double *dst = C[ii];  
        double *A_src = (double *)pointerA + f*A_SQ;      
        double *B_src = (double *)pointerB + f*B_SQ;     
        
        //        int XA0 = 0;
        //        int x = idx_x;
        //for (int x = 0; x < C_dims[1]; x++) 
        //{		
        
        //        XA0 = A_dims[0]*x;
        double *A_src2 =A_src+XA0; 
        // XA0+=A_dims[0];
        //        int y = idx_y;
        //for (int y = 0; y < C_dims[0]; y++) 
        //{
        double val = 0;
        double *A_off = A_src2+y;
        double *B_off = B_src;
        
        for (int xp = 0; xp < B_dims[1]; xp++) 
          {
            double *A_temp = A_off;						
            double *B_temp = B_off;	  
            for (int yp = 0; yp < B_dims[0]; yp++) 	  
              {
                val += *(A_temp++) * *(B_temp++);
              }
            
            A_off+=A_dims[0];
            B_off+=B_dims[0];
            
          }			 
        
        //*(dst + (x*C_dims[0] + y)) += val;		

        add_val += val;
        // }
        //}
      }

    *(dst + (idx_x*C_dims[0] + idx_y)) += add_val;

    
    //     if(ii==0 && idx_x==0 && idx_y==0 && level == interval){
    //       printf("sizeof(double) in GPU %llu\n", (unsigned long long int)sizeof(double));
    //       printf("sizeof(double*) in GPU %llu\n", (unsigned long long int)sizeof(double*));
    //       printf("sizeof(unsigned long int) in GPU %llu\n", (unsigned long long int)sizeof(unsigned long int));
    //       printf("sizeof(unsigned long long int) in GPU %llu\n", (unsigned long long int)sizeof(unsigned long long int));
    //       printf("%f\n", *(dst + (idx_x*C_dims[0] + idx_y)));
    //       printf("%f\n", C[0]);
    //       //       printf("%llu\n", (unsigned long long int)dst + (idx_x*C_dims[0] + idx_y));
    //       //       printf("%llu\n\n", (unsigned long long int)C);
    
    //     } 
  
  }

    
  return;
}



/********************************************/
/* function for calculating part */
/********************************************/
extern "C"
__global__
void
process_part(
 double *A,  
 double *B, 
 double *C, 
 int *A_dims_array, 
 int *B_dims_array, 
 int len,
 int interval, 
 int L_MAX,
 int *error_array,
 int error_array_num
) 
{


  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  // int ii = threadIdx.z;
  // int level = blockIdx.z; 
  int ii = blockIdx.z % len;
  int level = blockIdx.z / len; 

  int A_dims[3] = { A_dims_array[level*3], A_dims_array[level*3+1], A_dims_array[level*3+2] };
  int B_dims[3] = { B_dims_array[ii*3], B_dims_array[ii*3+1], B_dims_array[ii*3+2] };
  int C_dims[2] = { A_dims[0] - B_dims[0] + 1, A_dims[1] - B_dims[1] + 1 };


  if(0 <= ii && ii < len && 0 <= idx_x && idx_x < C_dims[1] && 0 <= idx_y && idx_y < C_dims[0] && 0 <= level && level < (L_MAX - interval) ) {

    int num_features = A_dims[2];
    const int A_SQ = A_dims[0]*A_dims[1];
    const int B_SQ = B_dims[0]*B_dims[1];
    double add_val = 0;

    int x = idx_x;
    int y = idx_y;
    int XA0 = A_dims[0]*x;
    
    /* apply loop condition */
    for(int i=0; i<error_array_num; i++){
      if(error_array[i] == level)
        return;
    }
    

    /* adjust the location of pointer of C */
    double *dst;
    unsigned long long int pointer = (unsigned long long int)C;
    for(int a=0; a<level; a++) {
      //for(int b=0; b<ii; b++){
      for(int b=0; b<len; b++){
        int height = A_dims_array[a*3] - B_dims_array[b*3] + 1;
        int width = A_dims_array[a*3 + 1] - B_dims_array[b*3 + 1] + 1;
        
        /* error semantics */
        if(height < 1 || width < 1){
          printf("Invalid input in GPU\n");
          return;
        }
        
        pointer += (unsigned long long int)(height*width*sizeof(double));
      }
    }

    for(int b=0; b<ii; b++){
      int height = A_dims_array[level*3] - B_dims_array[b*3] + 1;
      int width  = A_dims_array[level*3 + 1] - B_dims_array[b*3 + 1] + 1;

       /* error semantics */
        if(height < 1 || width < 1){
          printf("Invalid input in GPU\n");
          return;
        }

      pointer += (unsigned long long int)(height*width*sizeof(double));
    }
    

    dst = (double *)pointer;

    /* adjust the location of pointer of A */
    unsigned long long int pointerA = (unsigned long long int)A;
    for(int a=0; a<level; a++) {
      pointerA += (unsigned long long int)(A_dims_array[a*3]*A_dims_array[a*3 + 1]*A_dims_array[a*3 + 2]*sizeof(double));
    } 
    
    /* adjust the location of pointer of B */
    unsigned long long int pointerB = (unsigned long long int)B;
    for(int b=0; b<ii; b++) {
      pointerB += (unsigned long long int)(B_dims_array[b*3]*B_dims_array[b*3 + 1]*B_dims_array[b*3 + 2]*sizeof(double));
    } 
    
    for(int f = 0; f < num_features; f++) // num_features = 31
      {  
        //double *dst = C[ii];  
        double *A_src = (double *)pointerA + f*A_SQ;      
        double *B_src = (double *)pointerB + f*B_SQ;     
        
        //        int XA0 = 0;
        //        int x = idx_x;
        //for (int x = 0; x < C_dims[1]; x++) 
        //{		
        
        //        XA0 = A_dims[0]*x;
        double *A_src2 =A_src+XA0; 
        // XA0+=A_dims[0];
        //        int y = idx_y;
        //for (int y = 0; y < C_dims[0]; y++) 
        //{
        double val = 0;
        double *A_off = A_src2+y;
        double *B_off = B_src;
        
        for (int xp = 0; xp < B_dims[1]; xp++) 
          {
            double *A_temp = A_off;						
            double *B_temp = B_off;	  
            for (int yp = 0; yp < B_dims[0]; yp++) 	  
              {
                val += *(A_temp++) * *(B_temp++);
              }
            
            A_off+=A_dims[0];
            B_off+=B_dims[0];
            
          }			 
        
        //*(dst + (x*C_dims[0] + y)) += val;		
        add_val += val;
        // }
        //}
      }


    *(dst + (idx_x*C_dims[0] + idx_y)) += add_val;

  }
  
  return;
}
