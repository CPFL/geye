///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////Car tracking project with laser_radar_data_fusion/////////////////////////////////////////
//////////////////////////////////////////////////////////////////////Copyright 2009-10 Akihiro Takeuchi///////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////detect_func.h functions about object-detection (to extend detect.cc) /////////////////////////////////////////

#include <stdio.h>	
#include "MODEL_info.h"		//File information

#ifndef INCLUDED_DFunctions_
#define INCLUDED_DFunctions_

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//featurepyramid.cpp

//matrix initialization
double *ini_scales(Model_info *MI,IplImage *IM,int X,int Y);					//initialize scales
extern int *ini_featsize(Model_info *MI);								//initialize feature size

//feature calculatation
extern double **calc_f_pyramid(IplImage *Image,Model_info *MI,int *FTSIZE,double *scale);	


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//getboxes.cpp

//get boundary box coordinate 
extern double *get_boxes(double **features,double *scales,int *FSIZE,MODEL *MO,int *Dnum,double *A_SCORE,double thresh);

//release matrix
extern void free_features(double **features,Model_info *MI);		//release features

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//nms.cpp 
extern double *nms(double *boxes,double overlap,int *num,MODEL *MO);	//Non_maximum suppression function (extended to detect.cc)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//tracking.cpp 
extern RESULT *get_new_rects(IplImage *Image,MODEL *MO,double *boxes,int *NUM);	//get new_rectangle pixel_point


#endif