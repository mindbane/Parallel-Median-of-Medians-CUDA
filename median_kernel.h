#ifndef MEDIAN_KERNEL_H
#define MEDIAN_KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>

#define GREATER_THAN 1
#define EQUAL_TO 0
#define LESS_THAN -1
#define BLOCK_SIZE 256
#define GROUP_SIZE 8
#define RESULT_SIZE ((BLOCK_SIZE) / (GROUP_SIZE))

__global__ void FindMedians(float *, float *);
__global__ void PrefixScan(int *, int *, int *);
__global__ void CalcRelativePosition(float *, int *, float, int);
__global__ void ReorderElements(float *, float *, int *, float, int);

#endif
