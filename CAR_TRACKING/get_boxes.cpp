///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////Car tracking project with laser_radar_data_fusion/////////////////////////////////////////
//////////////////////////////////////////////////////////////////////Copyright 2009-10 Akihiro Takeuchi///////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////get_boxes.cpp  detect boundary-boxes-coordinate of oject /////////////////////////////////////////////////////

//C++ library
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Header files
#include "MODEL_info.h"				//File information
#include "get_boxes_func.h"			//external functions 
#include "Common.h"

#include "for_use_GPU.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//definiton of functions//

//subfunctions for detection

double *padarray(double *feature,int *size,int padx,int pady);		//padd zeros to image 
double *flip_feat(double *feat,int *size);							//filip feature order (to reduce calculation time)
int *get_gmpc(double *score,double thresh,int *ssize,int *GMN);		//get good matches

//calculate root rectangle-cooridnate 
double *rootbox(int x,int y,double scale,int padx,int pady,int *rsize);	

//calculate @art rectangle-cooridnate 
double *partbox(int x,int y,int ax,int ay,double scale,int padx,int pady,int *psize,int *lx,int *ly,int *ssize);

//calculate accumulated HOG detector score
void calc_a_score(double *ac_score,double *score,int *ssize,int *rsize,Model_info *MI,double scale);

//Object-detection function (extended to main)
double *get_boxes(double **features,double *scales,int *FSIZE,MODEL *MO,int *Dnum,double *A_SCORE,double thresh);

//release-functions
void free_rootmatch(double **rootmatch, MODEL *MO);
void free_partmatch(double **partmatch, MODEL *MO);
void free_boxes(double **boxes,int LofFeat);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//sub functions
//padd zeros to image 
double *padarray(double *feature,int *size,int padx,int pady)
{
  const int NEW_Y=size[0]+pady*2;
  const int NEW_X=size[1]+padx*2;
  const int L=NEW_Y*padx;
  const int SPL=size[0]+pady;
  const int M_S = sizeof(double)*size[0];
  
  CUresult res;
  double *new_feature;
  
  res = cuMemHostAlloc((void **)&new_feature, NEW_Y*NEW_X * size[2] * sizeof(double), CU_MEMHOSTALLOC_DEVICEMAP); // feature (pad-added)
  if(res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc(new_feature) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  memset(new_feature, 0, NEW_Y*NEW_X * size[2] * sizeof(double));  // zero clear
  

  double *P=new_feature;
  double *S=feature;
  for(int ii=0;ii<size[2];ii++)	//31
    {
      P+=L; //row-all
      for(int jj=0;jj<size[1];jj++)	
		{
          P+=pady;
          //memcpy_s(P,M_S,S,M_S);
          memcpy(P, S,M_S);
          S+=size[0];
          P+=SPL;
		}
      P+=L; //row-all
	}
  size[0]=NEW_Y;
  size[1]=NEW_X;
  
  return(new_feature);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//flip feat 
double *flip_feat(double *feat,int *size)
{
	const int ORD[31]={10,9,8,7,6,5,4,3,2,1,18,17,16,15,14,13,12,11,19,27,26,25,24,23,22,21,20,30,31,28,29};
	const int S_S =sizeof(double)*size[0];

#ifndef ORIGINAL
    CUresult res;
#if 0
    res = cuCtxPushCurrent(ctx);
    if(res != CUDA_SUCCESS){
      printf("cuCtxPushCurrent() failed: res = %s\n", conv(res));
      exit(1);
    }
#endif
#endif

#ifdef ORIGINAL
	double *fliped = (double*)calloc(size[0]*size[1]*size[2],sizeof(double)); // feature (pad-added)
#else
    double *fliped;
    res = cuMemHostAlloc((void **)&fliped, size[0]*size[1]*size[2]*sizeof(double), CU_MEMHOSTALLOC_DEVICEMAP);
    if(res != CUDA_SUCCESS){
      printf("cuMemHostAlloc(fliped) failed: res = %s\n", conv(res));
      exit(1);
    }

    memset(fliped, 0, size[0]*size[1]*size[2]*sizeof(double));

#endif

	double *D=fliped;
	int DIM = size[0]*size[1];
	for(int ii=0;ii<size[2];ii++)
	{
		double *S=feat+DIM*(*(ORD+ii));

		for(int jj=1;jj<=size[1];jj++)
		{
			double *P=S-*(size)*jj; //*(size)=size[0]
			//memcpy_s(D,S_S,P,S_S);
            memcpy(D, P,S_S);
			D+=*size;
			P+=*size;
		}
	}

#ifndef ORIGINAL
#if 0
    res = cuCtxPopCurrent(&ctx);
    if(res != CUDA_SUCCESS){
      printf("cuCtxPopCurrent(ctx) failed: res = %s\n", conv(res));
      exit(1);
    }
#endif
#endif

	return(fliped);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//get good-matched pixel-coordinate 
int *get_gmpc(double *score,double thresh,int *ssize,int *GMN)
{
  const int L = ssize[0]*ssize[1];
  double *P=score;
  int NUM=0;
  double MAX_SCORE=thresh;
  
  for(int ii=0;ii<L;ii++)
    {
      if(*P>thresh) 
        {
          NUM++;	
          if(*P>MAX_SCORE) MAX_SCORE=*P;
        }
      P++;
    }
  
  int *Out =(int*)malloc(NUM*2*sizeof(int));		
  P=score;
  
  int ii=0,qq=0;
  while(qq<NUM)
    {
      if(*P>thresh)
        {
          Out[2*qq]=ii%ssize[0];		//Y coordinate of match
          Out[2*qq+1]=ii/ssize[0];	//X coordinate of match
          qq++;
        }
      ii++;
      P++;
    }
  //printf("MAX_SCORE %f\n",MAX_SCORE);
  *GMN = NUM;
  return(Out);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//get root-box pixel coordinate 
double *rootbox(int x,int y,double scale,int padx,int pady,int *rsize)
{
	double *Out=(double*)malloc(sizeof(double)*4);
	Out[0]=((double)y-(double)pady+1)*scale;	//Y1
	Out[1]=((double)x-(double)padx+1)*scale;	//X1
	Out[2]=Out[0]+(double)rsize[0]*scale-1.0;	//Y2
	Out[3]=Out[1]+(double)rsize[1]*scale-1.0;	//X2
	return(Out);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//get part-box pixel coordinate 
//partbox(PA,x,y,ax[pp],ay[pp],scale,padx,pady,Pd_size,Ix[pp],Iy[pp],Isize);
double *partbox(int x,int y,int ax,int ay,double scale,int padx,int pady,int *psize,int *lx,int *ly,int *ssize)
{
	double *Out=(double*)malloc(sizeof(double)*4);
	int probex = (x-1)*2+ax;
	int probey = (y-1)*2+ay;
	int P = probey+probex*ssize[0];

	double px = (double)lx[P]+1.0;
	double py = (double)ly[P]+1.0;

	Out[0]=((py-2.0)/2.0+1.0-(double)pady)*scale;	//Y1
	Out[1]=((px-2.0)/2.0+1.0-(double)padx)*scale;	//X1
	Out[2]=Out[0]+(double)psize[0]*scale/2.0-1.0;		//Y2
	Out[3]=Out[1]+(double)psize[1]*scale/2.0-1.0;		//X2
	return(Out);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//calculate accumulated HOG detector score
void calc_a_score(double *ac_score,double *score,int *ssize,int *rsize,Model_info *MI,double scale)
{
	const int L_AS = MI->IM_HEIGHT*MI->IM_WIDTH;
	const int IHEI = MI->IM_HEIGHT;
	const int IWID = MI->IM_WIDTH;
	double sbin = (double)(MI->sbin);
	int pady_n = MI->pady;
	int padx_n = MI->padx;
	int block_pad = (int)(scale/2.0);

	int RY = (int)((double)rsize[0]*scale/2.0-1.0+block_pad);
	int RX = (int)((double)rsize[1]*scale/2.0-1.0+block_pad);

	for(int ii=0;ii<IWID;ii++)
	{
		int Xn=(int)((double)ii/scale+padx_n);

		for(int jj=0;jj<IHEI;jj++)
		{
			int Yn =(int)((double)jj/scale+pady_n);
			if(Yn<ssize[0] && Xn<ssize[1])
			{
				double sc = score[Yn+Xn*ssize[0]];		//get score of pixel

				int Im_Y = jj+RY;
				int Im_X = ii+RX;
				if(Im_Y<IHEI && Im_X<IWID)
				{
					double *PP=ac_score+Im_Y+Im_X*IHEI;		//consider root rectangle size
					if(sc>*PP) *PP=sc;						//save max score
				}
			}
		}
	}
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//detect boundary box 
double *get_boxes(double **features,double *scales,int *FSIZE,MODEL *MO,int *Dnum,double *A_SCORE,double thresh)
{
  
  //constant parameters
  const int max_scale = MO->MI->max_scale;
  const int interval = MO->MI->interval;
  const int sbin = MO->MI->sbin;
  const int padx = MO->MI->padx;
  const int pady = MO->MI->pady;
  const int NoR = MO->RF->NoR;
  const int NoP = MO->PF->NoP;
  const int NoC = MO->MI->numcomponent;
  const int *numpart = MO->MI->numpart;
  const int LofFeat=(max_scale+interval)*NoC;
  const int L_MAX = max_scale+interval;
  
  int **RF_size = MO->RF->root_size;
  int *rootsym = MO->RF->rootsym;	
  int *part_sym = MO->PF->part_sym;	
  int **part_size = MO->PF->part_size;
  double **rootfilter = MO->RF->rootfilter;
  double **partfilter=MO->PF->partfilter;
  int **psize = MO->MI->psize;
  
  //int *rm_size = (int*)malloc(sizeof(int)*NoC*2);				//size of root-matching-score-matrix 	
  //int *pm_size = (int*)malloc(sizeof(int)*NoP*2);				//size of part-matching-score-matrix 
  int **rm_size_array = (int **)malloc(sizeof(int *)*L_MAX);
  int **pm_size_array = (int **)malloc(sizeof(int *)*L_MAX);

  double **Tboxes=(double**)calloc(LofFeat,sizeof(double*));		//box coordinate information(Temp)
  int  *b_nums =(int*)calloc(LofFeat,sizeof(int));				//length of Tboxes 
  int count = 0;													
  int D_NUMS=0;													//number of detected boundary box
  CUresult res;
  
  /* matched score (root and part) */
  double ***rootmatch,***partmatch;  

  int *new_PADsize;  // need new_PADsize[L_MAX*3]
  size_t SUM_SIZE_feat = 0;

  double **featp2 = (double **)malloc(L_MAX*sizeof(double *));
  if(featp2 == NULL) {  // error semantics
    printf("allocate featp2 failed\n");
    exit(1);
  }


  /* allocate required memory for new_PADsize */
  new_PADsize = (int *)malloc(L_MAX*3*sizeof(int));
  if(new_PADsize == NULL) { // error semantics
    printf("allocate new_PADsize failed\n");
    exit(1);
  }
  
  
  /*******************************************************************/
  /* do padarray once and reuse it at calculating root and part time */

  /* calculate sum of size of padded feature */   
  for(int tmpL=0; tmpL<L_MAX; tmpL++) {
    
    int PADsize[3] = { FSIZE[tmpL*2], FSIZE[tmpL*2+1], 31 };
    int NEW_Y = PADsize[0] + pady*2;
    int NEW_X = PADsize[1] + padx*2;
    SUM_SIZE_feat += (NEW_X*NEW_Y*PADsize[2])*sizeof(double);

  }

  /* allocate region for padded feat in a lump */
  double *dst_feat;
  res = cuMemHostAlloc((void **)&dst_feat, SUM_SIZE_feat, CU_MEMHOSTALLOC_DEVICEMAP);
  if(res != CUDA_SUCCESS) {
    printf("cuMemHostAlloc(dst_feat) failed: res = %s\n", conv(res));
    exit(1);
  }

  memset(dst_feat, 0, SUM_SIZE_feat);  // zero clear

  /* distribute allocated region */
  unsigned long long int pointer_feat = (unsigned long long int)dst_feat;
  for(int tmpL=0; tmpL<L_MAX; tmpL++) {

    featp2[tmpL] = (double *)pointer_feat;
    int PADsize[3] = { FSIZE[tmpL*2], FSIZE[tmpL*2+1], 31 };
    int NEW_Y = PADsize[0] + pady*2;
    int NEW_X = PADsize[1] + padx*2;
    pointer_feat += (unsigned long long int)(NEW_X*NEW_Y*PADsize[2]*sizeof(double));

  }

  /* copy feat to feat2 */
  for(int tmpL=0; tmpL<L_MAX; tmpL++) {

    int PADsize[3] = { FSIZE[tmpL*2], FSIZE[tmpL*2+1], 31 };
    int NEW_Y = PADsize[0] + pady*2;
    int NEW_X = PADsize[1] + padx*2;
    int L = NEW_Y*padx;
    int SPL = PADsize[0] + pady;
    int M_S = sizeof(double)*PADsize[0];
    double *P = featp2[tmpL];
    double *S = features[tmpL];

    for(int ii=0; ii<PADsize[2]; ii++) 
      {
        P += L; 
        for(int jj=0; jj<PADsize[1]; jj++)
          {
            P += pady;
            memcpy(P, S, M_S);
            S += PADsize[0];
            P += SPL;
          }
        P += L;
      }

    new_PADsize[tmpL*3] = NEW_Y;
    new_PADsize[tmpL*3 + 1] = NEW_X;
    new_PADsize[tmpL*3 + 2] = PADsize[2];

  }

  /* do padarray once and reuse it at calculating root and part time */
  /*******************************************************************/





  /* allocation in a lump */
  int *dst_rm_size = (int *)malloc(sizeof(int)*NoC*2*L_MAX);
  if(dst_rm_size == NULL) {
    printf("allocate dst_rm_size failed\n");
    exit(1);
  }

  /* distribution to rm_size_array[L_MAX] */
  unsigned long long int ptr = (unsigned long long int)dst_rm_size;
  for(int i=0; i<L_MAX; i++) {
    rm_size_array[i] = (int *)ptr;
    ptr += (unsigned long long int)(NoC*2*sizeof(int));
  }

  /* allocation in a lump */
  int *dst_pm_size = (int *)malloc(sizeof(int)*NoP*2*L_MAX);  
  if(dst_pm_size == NULL) {
    printf("allocate dst_pm_size failed\n");
    exit(1);
  }

  /* distribution to pm_size_array[L_MAX] */
  ptr = (unsigned long long int)dst_pm_size;
  for(int i=0; i<L_MAX; i++) {
    pm_size_array[i] = (int *)ptr;
    ptr += (unsigned long long int)(NoP*2*sizeof(int));
  }

  
  ///////level
  
  
  for (int level=interval; level<L_MAX; level++)  // feature's loop(A's loop) 1level 1picture
    {
      /* parameters (related for level) */
      int L=level-interval;
      /* matched score size matrix */
      double scale=(double)sbin/scales[level];
      
      /**************************************************************************/      
      /* loop conditon */
      if(FSIZE[level*2]+2*pady<MO->MI->max_Y ||(FSIZE[level*2+1]+2*padx<MO->MI->max_X))
		{
          Tboxes[count]=NULL;
          count++;
          continue;
		}
      /* loop conditon */
      /**************************************************************************/      


    }  //for (level)  // feature's loop(A's loop) 1level 1picture

  
  /* allocate GPU memory and transfer feat, new_PADsize to GPU  */
  CUdeviceptr featp2_dev, new_PADsize_dev;
  res = cuMemAlloc(&featp2_dev, SUM_SIZE_feat);
  if(res != CUDA_SUCCESS) {
    printf("cuMemAlloc(featp2_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemAlloc(&new_PADsize_dev, L_MAX*3*sizeof(int));
  if(res != CUDA_SUCCESS) {
    printf("cuMemAlloc(new_PADsize_dev) failed: res = %s\n", conv(res));
    exit(1);
  }


  res = cuMemcpyHtoD(featp2_dev, featp2[0], SUM_SIZE_feat);
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD(featp2) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemcpyHtoD(new_PADsize_dev, new_PADsize, L_MAX*3*sizeof(int));
  if(res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD(new_PADsize) failed: res = %s\n", conv(res));
    exit(1);
  }



  ///////root calculation/////////
  /* calculate model score (only root) */
  rootmatch = fconvsMT_GPU(featp2_dev, rootfilter, rootsym, 1, NoR, new_PADsize, new_PADsize_dev, RF_size, rm_size_array, L_MAX, interval, FSIZE, padx, pady, MO->MI->max_X, MO->MI->max_Y, ROOT);   
  
  
  ///////part calculation/////////
  if(NoP>0)
    {
      /* calculate model score (only part) */
      partmatch = fconvsMT_GPU(featp2_dev, partfilter, part_sym, 1, NoP, new_PADsize, new_PADsize_dev, part_size, pm_size_array, L_MAX, interval, FSIZE, padx, pady, MO->MI->max_X, MO->MI->max_Y, PART);
    }
  

  /* free GPU memory for feat, new_PADsize  */      
  res = cuMemFree(featp2_dev);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFree(featp2_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

  res = cuMemFree(new_PADsize_dev);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFree(new_PADsize_dev) failed: res = %s\n", conv(res));
    exit(1);
  }

    
  count = 0;													
  D_NUMS=0;	
  for (int level=interval; level<L_MAX; level++)  // feature's loop(A's loop) 1level 1picture
    {
      /* parameters (related for level) */
      int L=level-interval;
      /* matched score size matrix */
      double scale=(double)sbin/scales[level];

      /**************************************************************************/      
      /* loop conditon */

      if(FSIZE[level*2]+2*pady<MO->MI->max_Y ||(FSIZE[level*2+1]+2*padx<MO->MI->max_X))
        {
          Tboxes[count]=NULL;
          count++;
          continue;
        }
    
      /* loop conditon */
      /**************************************************************************/
      
      ////nucom
      /* combine root and part score and detect boundary box for each-component */
      for(int jj=0;jj<NoC;jj++)
        {
          /* root score + offset */
          int RL = rm_size_array[level][jj*2]*rm_size_array[level][jj*2+1];  //length of root-matching 
          int RI = MO->MI->ridx[jj];  //root-index
          int OI = MO->MI->oidx[jj];  //offset-index
          int RL_S =sizeof(double)*RL;
          
          double OFF = MO->MI->offw[RI];  //offset information
          double *SCORE = (double*)malloc(RL_S);  //Matching score matrix
          
          /* add offset */
          //memcpy_s(SCORE,RL_S,rootmatch[jj],RL_S);
          memcpy(SCORE, rootmatch[level][jj], RL_S);
          double *SC_S = SCORE;
          double *SC_E = SCORE+RL;
          while(SC_S<SC_E) *(SC_S++)+=OFF;
          
          /* root matching size */
          int R_S[2]={rm_size_array[level][jj*2], rm_size_array[level][jj*2+1]};
          int SNJ =sizeof(int*)*numpart[jj];
          
          
          /* anchor matrix */
          int *ax = (int*)malloc(SNJ);
          int *ay = (int*)malloc(SNJ);
          
          /* boudary index */
          int **Ix =(int**)malloc(SNJ);
          int **Iy =(int**)malloc(SNJ);
          
          
          /* add parts */
          if(NoP>0)
            {			
              
              for (int kk=0;kk<numpart[jj];kk++)
                {
                  int DIDX = MO->MI->didx[jj][kk];
                  int DID_4 = DIDX*4;
                  int PIDX = MO->MI->pidx[jj][kk];
                  
                  /* anchor */
                  ax[kk] = MO->MI->anchor[DIDX*2]+1;
                  ay[kk] = MO->MI->anchor[DIDX*2+1]+1;
                  
                  /* set part-match */
                  double *match = partmatch[L][PIDX];
                  
                  /* size of part-matching */
                  int PSSIZE[2] ={pm_size_array[L][PIDX*2], pm_size_array[L][PIDX*2+1]};
                  
                  double *Q = match;
                  for(int ss=0;ss<PSSIZE[0]*PSSIZE[1];ss++) *(Q++)*=-1;
                  
                  /* index matrix */
                  Ix[kk] =(int*)malloc(sizeof(int)*PSSIZE[0]*PSSIZE[1]);
                  Iy[kk] =(int*)malloc(sizeof(int)*PSSIZE[0]*PSSIZE[1]);
                  
                  /* decide position of part for all pixels */
                  double *M = dt(match, MO->MI->def[DID_4], MO->MI->def[DID_4+1], MO->MI->def[DID_4+2], MO->MI->def[DID_4+3], PSSIZE, Ix[kk], Iy[kk]);
                  
                  /* add part score */
                  add_part_calculation(SCORE, M, R_S, PSSIZE, ax[kk], ay[kk]); 

                  s_free(M);
                }
            }
          
          /* get all good matches */
          int GMN;
          int *GMPC = get_gmpc(SCORE,thresh,R_S,&GMN);
          int RSIZE[2]={MO->MI->rsize[jj*2],MO->MI->rsize[jj*2+1]};
          
          int GL = (numpart[jj]+1)*4+3;  //31
          
          /* calculate accumulated score */
          calc_a_score(A_SCORE, SCORE, R_S, RSIZE, MO->MI, scale);
          
          /* detected box coordinate(current level) */
          double *t_boxes = (double*)calloc(GMN*GL,sizeof(double));
          
          for(int kk=0;kk<GMN;kk++)
            {
              double *P_temp = t_boxes+GL*kk;
              int y = GMPC[2*kk];
              int x = GMPC[2*kk+1];
              
              /* calculate root box coordinate */
              double *RB =rootbox(x,y,scale,padx,pady,RSIZE);
              //memcpy_s(P_temp,sizeof(double)*4,RB,sizeof(double)*4);
              memcpy(P_temp, RB,sizeof(double)*4);
              s_free(RB);
              P_temp+=4;
              
              for(int pp=0;pp<numpart[jj];pp++)
                {
                  int PBSIZE[2]={psize[jj][pp*2],psize[jj][pp*2+1]};
                  int Isize[2]={pm_size_array[L][MO->MI->pidx[jj][pp]*2], pm_size_array[L][MO->MI->pidx[jj][pp]*2+1]};
                  
                  /* calculate part box coordinate */
                  double *PB = partbox(x,y,ax[pp],ay[pp],scale,padx,pady,PBSIZE,Ix[pp],Iy[pp],Isize);
                  //memcpy_s(P_temp,sizeof(double)*4,PB,sizeof(double)*4);
                  memcpy(P_temp, PB,sizeof(double)*4);
                  P_temp+=4;
                  s_free(PB);
                }
              /* component number and score */
              *(P_temp++)=(double)jj;					//component number 
              *(P_temp++)=SCORE[x*R_S[0]+y];			//score of good match
              *P_temp = scale;
            }
          
          /* save box information */
          if(GMN>0) Tboxes[count]=t_boxes;
          else Tboxes[count]=NULL;
          b_nums[count]=GMN;
          count++;
          D_NUMS+=GMN;			//number of detected box 
          
          /* release */
          s_free(GMPC);
          s_free(SCORE);
          s_free(ax);
          s_free(ay);
          for(int ss=0;ss<numpart[jj];ss++)
            {
              s_free(Ix[ss]);
              s_free(Iy[ss]);
            }
          s_free(Ix);		
          s_free(Iy);
          
        }
      ////numcom
      
    }
  ////level
  
  
  /* free memory regions */
  res = cuMemFreeHost((void *)featp2[0]);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFreeHost(featp2[0]) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  s_free(featp2);
  
  
  res = cuMemFreeHost((void *)rootmatch[interval][0]);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFreeHost(rootmatch[0][0]) failed: res = %s\n", conv(res));
    exit(1);
  }
  s_free(rootmatch[0]);
  s_free(rootmatch);
  
  
  res = cuMemFreeHost((void *)partmatch[0][0]);
  if(res != CUDA_SUCCESS) {
    printf("cuMemFreeHost(partmatch[0][0]) failed: res = %s\n", conv(res));
    exit(1);
  }
  
  s_free(partmatch[0]);
  s_free(partmatch);
  
  s_free(new_PADsize);
  
  
  
  /* release */
  s_free(rm_size_array[0]);
  s_free(rm_size_array);
  s_free(pm_size_array[0]);
  s_free(pm_size_array);
  





  /* Output boundary-box coorinate information */
  int GL=(numpart[0]+1)*4+3;
  double *boxes=(double*)calloc(D_NUMS*GL,sizeof(double));		//box coordinate information(Temp)
  double *T1=boxes;
  int cc=0;
  
  
  for(int ii=0;ii<LofFeat;ii++)
	{
      int num_t = b_nums[ii]*GL;
      double *T2 = Tboxes[ii];
      if(num_t>0)
		{
          //memcpy_s(T1,sizeof(double)*num_t,T2,sizeof(double)*num_t);
          memcpy(T1, T2,sizeof(double)*num_t);
          T1+=num_t;
		}
	}
  
  
  double AS_OFF = abs(thresh);
  
  /////////////////////////////////
  /* accumulated score calculation */
  double max_ac = 0.0;
  
  
  /* add offset to accumulated score */
  for(int ii=0;ii<MO->MI->IM_HEIGHT*MO->MI->IM_WIDTH;ii++)
	{
      if(A_SCORE[ii]<thresh) A_SCORE[ii]=0.0;
      else
		{
          A_SCORE[ii]+=AS_OFF;
          if(A_SCORE[ii]>max_ac) max_ac=A_SCORE[ii];
		}
	}
  /* normalization */
  if(max_ac>0.0)
	{
      double ac_ratio = 1.0/max_ac;
      for(int ii=0;ii<MO->MI->IM_HEIGHT*MO->MI->IM_WIDTH;ii++){A_SCORE[ii]*=ac_ratio;}
	}
  
  /* release */
  free_boxes(Tboxes,LofFeat);
  s_free(b_nums);
  
  /* output result */
  *Dnum=D_NUMS;
  
  
  return(boxes);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////release functions //////

//free root-matching result 

void free_rootmatch(double **rootmatch, MODEL *MO)
{
  CUresult res;


  if(rootmatch!=NULL)
	{
      for(int ii=0;ii<MO->RF->NoR;ii++)
		{	
#ifdef ORIGINAL
          s_free(rootmatch[ii]);
#else
          res = cuMemFreeHost((void *)rootmatch[ii]);
          if(res != CUDA_SUCCESS){
            printf("cuMemFreeHost(rootmatch) failed: res = %s\n", conv(res));
            exit(1);
          }
#endif
		}
      s_free(rootmatch);
	}
  

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//free part-matching result 

void free_partmatch(double **partmatch, MODEL *MO)
{
  CUresult res;

  if(partmatch!=NULL)
	{
      for(int ii=0;ii<MO->PF->NoP;ii++)
		{
#ifdef ORIGINAL
          s_free(partmatch[ii]);
#else
          res = cuMemFreeHost((void *)partmatch[ii]);
          if(res != CUDA_SUCCESS){
            printf("cuMemFreeHost(partmatch) failed: res = %s\n", conv(res));
            exit(1);
          }
#endif
		}
      s_free(partmatch);
	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//free detected boxes result 

void free_boxes(double **boxes, int LofFeat)
{
	if(boxes!=NULL)
	{
		for(int ii=0;ii<LofFeat;ii++)
		{
			s_free(boxes[ii]);
		}
		s_free(boxes);
	}
}
