#include "median_kernel.h"

__global__ void FindMedians(float *input, float *output)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int groupPosition = threadIdx.x % GROUP_SIZE;
	int groupId = threadIdx.x / GROUP_SIZE;
	
	//Track the number of INFINITY padding the array
	__shared__ int numNumbers[RESULT_SIZE];
	if (groupPosition == 0)
	{
		numNumbers[groupId] = GROUP_SIZE;
	}

	//Find the median of each group
	int i, j;
	for(i=0; i < GROUP_SIZE-1; i++)
	{
		if (groupPosition == i)
		{
			if (input[id] == 1/0.) // 1/0. == INFINITY
			{
				numNumbers[groupId]--;
			}
			else
			{
				//Simple bubble sort
				int smallIndex = 0;
				for(j=1; j < GROUP_SIZE-groupPosition; j++)
				{
					if (input[id + smallIndex] > input[id+j])
					{
						smallIndex = j;
					}
				}
				float temp = input[id];
				input[id] = input[id + smallIndex];
				input[id + smallIndex] = temp;
			}
		}
		__syncthreads();
	}

	//determine which element has the median in it ignoring infinities
	int medianPosition = (numNumbers[groupId] + 1) / 2 - 1;
	if (medianPosition < 0)
	{
		medianPosition = 0;
	}

	//write that to the new array of medians
	if (groupPosition == medianPosition)
	{
		output[blockIdx.x * RESULT_SIZE + groupId] = input[id];
	}
}

__global__ void PrefixScan(int *input, int *output, int *blockOffset)
{
	int id = threadIdx.x;
	
	//We are always operating on 256 or less chunks at a time.
	__shared__ int swap[BLOCK_SIZE];
	
	int offset = 1;
	swap[2*id] = input[2*id];
	swap[2*id+1] = input[2*id+1];
	
	 // build sum in place up the tree
	int i;
	for (i = BLOCK_SIZE/2; i > 0; i /= 2)
	{
		__syncthreads();
	
		if (id < i)
		{
			int indexA = offset*(2*id+1)-1;
			int indexB = offset*(2*id+2)-1;
			swap[indexB] += swap[indexA];
		}
		offset *= 2;
	}
	
	int curBlockOffset = *blockOffset;
	if (id == 0)
	{
		*blockOffset += swap[BLOCK_SIZE-1];
		swap[BLOCK_SIZE-1] = 0;
	}
	
	 // traverse down tree & build scan
	for (i=1; i < BLOCK_SIZE; i *= 2)
	{
		offset /= 2;
		__syncthreads();
		
		if (id < i)
		{
			int indexA = offset*(2*id+1)-1;
			int indexB = offset*(2*id+2)-1;
			int temp = swap[indexA];

			swap[indexA] = swap[indexB];
			swap[indexB] += temp;
		}
	}
	
	__syncthreads();
	
	output[2*id] = curBlockOffset + swap[2*id];
	output[2*id+1] = curBlockOffset + swap[2*id+1];
}

__global__ void CalcRelativePosition(float *input, int *compareResult, float medianValue, int compareType)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	//Determine position relative to median
	switch(compareType)
	{
		case LESS_THAN:
			if (input[id] < medianValue)
			{
				compareResult[id] = 1;
			}
			else
			{
				compareResult[id] = 0;
			}
			break;
		case EQUAL_TO:
			if (input[id] == medianValue)
			{
				compareResult[id] = 1;
			}
			else
			{
				compareResult[id] = 0;
			}
			break;
		case GREATER_THAN:
			if (input[id] > medianValue)
			{
				compareResult[id] = 1;
			}
			else
			{
				compareResult[id] = 0;
			}
			break;
	}
}

__global__ void ReorderElements(float *input, float *output, int *threadOffset, float medianValue, int compareType)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	switch(compareType)
	{
		case LESS_THAN:
			if (input[id] >= medianValue)
			{
				return;
			}
			break;
		case EQUAL_TO:
			if (input[id] != medianValue)
			{
				return;
			}
			break;
		case GREATER_THAN:
			if (input[id] <= medianValue)
			{
				return;
			}
			break;
	}
	
	output[threadOffset[id]] = input[id];
}
