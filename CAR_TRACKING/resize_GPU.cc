///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////Car tracking project with laser_radar_data_fusion/////////////////////////////////////////
//////////////////////////////////////////////////////////////////////Copyright 2009-10 Akihiro Takeuchi///////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////resize.cpp   resize image (Input and Output must be double-array) ////////////////////////////////////////////

//C++ library
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
//ORIGINAL header files
#include "Common.h"

#include "switch_float.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//main function
FLOAT *resize(FLOAT *src,int *sdims,int *odims,FLOAT scale);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define USE_PTHREAD

typedef struct {
  FLOAT *src_top;
  int *src_size;
  FLOAT *dst_top;
  int *dst_size;
}resize_thread_arg;


/*********************************************/
/* sub function to get pixel values from src */
/*********************************************/
static inline 
FLOAT getPixelVal(FLOAT *src, int x, int y, int width, int height)
{
  int access_x = (x < 0) ? 0 : 
    (x < width) ? x : (width-1);

  int access_y = (y < 0) ? 0 : 
    (y < height) ? y : (height-1);

  return src[access_x*height + access_y];

}


/***************************************************************/
/* image resizing function using bilinear interpolation method */
/***************************************************************/
void *bilinear_resizing(void *arg)
{
  resize_thread_arg *this_arg = (resize_thread_arg *)arg;
  FLOAT *src_top  = this_arg->src_top;
  int   *src_size = this_arg->src_size;
  FLOAT *dst_top  = this_arg->dst_top;
  int   *dst_size = this_arg->dst_size;

  const int src_height = src_size[0];
  const int src_width  = src_size[1];
  const int dst_height = dst_size[0];
  const int dst_width  = dst_size[1];

  const FLOAT hfactor = (FLOAT)src_height/dst_height;
  const FLOAT wfactor = (FLOAT)src_width/dst_width;

  for (int channel = 0; channel<dst_size[2]; channel++)
    {
      /*
        The function "Ipl_to_FLOAT"(defined in featurepyramid.cpp)
        break input image down by each color channel. 
        So, we have to adjust the pointer location to refer 
        each color values.
      */
      FLOAT *src = src_top + channel*src_height*src_width;
      FLOAT *dst = dst_top + channel*dst_height*dst_width;
          
      for (int dst_x=0; dst_x<dst_width; dst_x++)
        {
          /* pixel position on "src" correspond to "dst"(true value) */
          FLOAT src_x_decimal = wfactor * (FLOAT)dst_x;
          /* pixel position on "src" correspond to "dst"(truncated integer value) */
          int   src_x         = (int)src_x_decimal;
          
          for (int dst_y=0; dst_y<dst_height; dst_y++)
            {
              /* pixel position on "src" correspond to "dst"(true value) */
              FLOAT src_y_decimal = hfactor * (FLOAT)dst_y;
              /* pixel position on "src" correspond to "dst"(truncated integer value) */
              int   src_y         = (int)src_y_decimal;
              
              /* bilinear interpolation */
              FLOAT src_val[4] = {
                (FLOAT)getPixelVal(src, src_x, src_y, src_width, src_height),
                (FLOAT)getPixelVal(src, src_x+1, src_y, src_width, src_height),
                (FLOAT)getPixelVal(src, src_x, src_y+1, src_width, src_height),
                (FLOAT)getPixelVal(src, src_x+1, src_y+1, src_width, src_height)
              };
              
              FLOAT c_element[4] = {0};
              FLOAT newval = 0;
              
              for (int i=0; i<4; i++)
                {
                  c_element[i] = (FLOAT)((unsigned int)src_val[i] & 0xffffffff);
                }
              
              FLOAT new_element = (FLOAT)(
                                          (src_x + 1 - src_x_decimal)*(src_y + 1 - src_y_decimal)*c_element[0] + 
                                          (src_x_decimal - src_x)*(src_y + 1 - src_y_decimal)*c_element[1] + 
                                          (src_x + 1 - src_x_decimal)*(src_y_decimal - src_y)*c_element[2] + 
                                          (src_x_decimal - src_x)*(src_y_decimal - src_y)*c_element[3]
                                          );
              
              dst[dst_x*dst_height + dst_y] = (FLOAT)(new_element);
            }
        }
    }

  return (void *)NULL;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**************************/
/* main function (resize) */
/**************************/
void resize_byGPU(FLOAT *org_image, 
                  int *org_image_size,
                  FLOAT **resized_image,
                  int *resized_image_size, 
                  int interval,
                  int LEN)
{
  
#ifdef USE_PTHREAD  
  /* pthread handler */
  /* to calculate all resized image, the required number of threads is (LEN - interval) */
  pthread_t *thread = (pthread_t *)calloc(LEN - interval, sizeof(pthread_t));
#endif
  
  /* structure to carry data to pthread function */
  resize_thread_arg *args = (resize_thread_arg *)calloc(LEN - interval, sizeof(resize_thread_arg));
  int thread_count = 0;
    
  /* calculate sum size of resized image */
  int sum_size_image = 0;
  for(int level=0; level<LEN; level++)
    {    
      sum_size_image += resized_image_size[level*3] * 
        resized_image_size[level*3 + 1] * 
        resized_image_size[level*3 + 2];
    }
  
  /* allocate memory region for resized image */
  FLOAT *resized_image_dst = (FLOAT *)calloc(sum_size_image, sizeof(FLOAT));
  
  /* distribute memory region to each resized image */
  unsigned long long int ptr_resized_image = (unsigned long long int)resized_image_dst;
  for (int level=0; level<LEN; level++)
    {
      resized_image[level] = (FLOAT *)(ptr_resized_image);
      ptr_resized_image += resized_image_size[level*3] * resized_image_size[level*3 + 1] * resized_image_size[level*3 + 2] * sizeof(FLOAT);
    }
  
  /* resizing */
  for (int level=0; level<interval; level++)
    {
      /* assign data for pthread function */
      args[thread_count].src_top  = org_image;
      args[thread_count].src_size = org_image_size;
      args[thread_count].dst_top  = resized_image[level];
      args[thread_count].dst_size = &resized_image_size[level*3];
      
#ifdef USE_PTHREAD
      pthread_create(&thread[thread_count], NULL, bilinear_resizing, (void *)&args[thread_count]);
      thread_count++;
#else
      bilinear_resizing((void *)&args[thread_count]);
#endif
    }
  
  
  /* extra resizing */
  for (int level=2*interval; level<LEN; level++)
    {
      /* assign data for pthread function */
      args[thread_count].src_top  = org_image;
      args[thread_count].src_size = org_image_size;
      args[thread_count].dst_top  = resized_image[level];
      args[thread_count].dst_size = &resized_image_size[level*3];
      
#ifdef USE_PTHREAD
      pthread_create(&thread[thread_count], NULL, bilinear_resizing, (void *)&args[thread_count]);
      thread_count++;
#else
      bilinear_resizing((void *)&args[thread_count]);
#endif
    }
  
#ifdef USE_PTHREAD
  /* wait for all pthread complete its work */
  for (int counter=0; counter<LEN-interval; counter++)
    {
      pthread_join(thread[counter], NULL);
    }
#endif
  
  /* (interval <= level < 2*interval) use same resize scale as (0 <= level < interval) */
  for (int level=interval; level<2*interval; level++)
    {
      int copy_size = resized_image_size[level*3] * resized_image_size[level*3 + 1] * resized_image_size[level*3 + 2] * sizeof(FLOAT);
      memcpy(resized_image[level], resized_image[level-interval], copy_size);
    }
  
  /* cleanup */
#ifdef USE_PTHREAD
  free(thread);
#endif
  free(args);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/********************************************/
/* calculate each image size after resizing */
/********************************************/
void calc_resized_image_size(int *org_image_size, 
                             int *resized_image_size,
                             int interval, 
                             FLOAT sc,
                             int max_scale, 
                             FLOAT *scale_array)
{
 const int org_height    = org_image_size[0];
 const int org_width     = org_image_size[1];
 const int org_nChannels = org_image_size[2];

  for (int ii=0; ii<interval; ii++)
    {
      /* resizing rate */
      FLOAT scale = 1.0/pow(sc, ii);

      /* calculte and assign resized image size */
      if (scale==1.0)
        {
          resized_image_size[ii*3]     = org_height;
          resized_image_size[ii*3 + 1] = org_width;
          resized_image_size[ii*3 + 2] = org_nChannels;
        }
      else
        {
          resized_image_size[ii*3]     = (int)((FLOAT)org_height*scale + 0.5);
          resized_image_size[ii*3 + 1] = (int)((FLOAT)org_width*scale + 0.5);
          resized_image_size[ii*3 + 2] = org_nChannels;
        }

      memcpy(resized_image_size + (ii+interval)*3, resized_image_size + (ii)*3, 3*sizeof(int));

      /* save scale */
      scale_array[ii] = scale*2;
      scale_array[ii+interval] = scale;

      /* extra resizing  */
      const FLOAT extra_scale = 0.5;
      for (int jj=ii+interval; jj<max_scale; jj+=interval)
        {
          resized_image_size[(jj+interval)*3]     = (int)((FLOAT)resized_image_size[jj*3]*extra_scale + 0.5);
          resized_image_size[(jj+interval)*3 + 1] = (int)((FLOAT)resized_image_size[jj*3 + 1]*extra_scale + 0.5);
          resized_image_size[(jj+interval)*3 + 2] = resized_image_size[jj*3 + 2];
          
          /* save scale */
          scale_array[jj+interval] = 0.5*scale_array[jj];
        }
    }

  return;

}
