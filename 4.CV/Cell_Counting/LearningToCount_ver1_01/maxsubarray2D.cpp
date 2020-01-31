/*
Simple implementation of the 2D max subarray algorithm. Faster implementations are possible.
Code copyright:   Victor Lempitsky, 2011
*/


#include <mex.h>
#include <matrix.h>
#include <memory.h>
#include <math.h>


double *integralImage;
double bestSum;
int bestMinY, bestMaxY;
int bestMinX, bestMaxX;

void ComputeIntegralImage(double *a, int width, int height)
{
	memset(integralImage, 0, sizeof(double)*height);

	for(int x = 1; x <= width; x++)
		for(int y = 0; y < height; y++)
			integralImage[x*height+y] = integralImage[(x-1)*height+y]+a[(x-1)*height+y];
}

void max_subarray_cols(int colMin, int colMax, int width, int height)
{
	int cur_start = 0;
	double cur_sum = 0;

	for(int i = 0; i < height; i++)
	{
		cur_sum += integralImage[(colMax+1)*height+i]-integralImage[colMin*height+i];
		if(cur_sum > bestSum)
		{
			bestMinX = colMin+1; //adding one to adopt MATLAB's convention
			bestMaxX = colMax+1;
			bestMinY = cur_start+1;
			bestMaxY = i+1;
			bestSum = cur_sum;
		}
		if(cur_sum < 0)
		{
			cur_start = i+1;
			cur_sum = 0;
		}
	}
}


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) 
{
 	if (nrhs != 1 || !mxIsDouble(prhs[0]) || mxGetNumberOfDimensions (prhs[0]) != 2 ||	mxIsComplex (prhs[0]))
	{
	    mexErrMsgTxt ("Invalid input.\n");
	    return;
	}

	int rows = mxGetM (prhs[0]);
	int cols = mxGetN (prhs[0]);
	if(nlhs > 0)	plhs[0] = mxCreateNumericMatrix (1, 1, mxINT32_CLASS, mxREAL);
	if(nlhs > 1)	plhs[1] = mxCreateNumericMatrix (1, 1, mxINT32_CLASS, mxREAL);
	if(nlhs > 2)	plhs[2] = mxCreateNumericMatrix (1, 1, mxINT32_CLASS, mxREAL);
	if(nlhs > 3)	plhs[3] = mxCreateNumericMatrix (1, 1, mxINT32_CLASS, mxREAL);
	if(nlhs > 4)	plhs[4] = mxCreateNumericMatrix (1, 1, mxDOUBLE_CLASS, mxREAL);

	integralImage = new double[rows*(cols+1)];
	ComputeIntegralImage(mxGetPr(prhs[0]),cols,rows);

	bestSum = -1e30;
	for(int i = 0; i < cols; i++)
		for(int j = i; j < cols; j++)
			max_subarray_cols(i, j, cols, rows);

	delete[] integralImage;

	if(nlhs > 0)	*(int *)mxGetData(plhs[0]) = bestMinY;
	if(nlhs > 1)	*(int *)mxGetData(plhs[1]) = bestMaxY;
	if(nlhs > 2)	*(int *)mxGetData(plhs[2]) = bestMinX;
	if(nlhs > 3)	*(int *)mxGetData(plhs[3]) = bestMaxX;
	if(nlhs > 4)	*mxGetPr(plhs[4]) = bestSum;
}