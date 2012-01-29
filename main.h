#ifndef MAIN_H
#define MAIN_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <cutil.h>
#include "median_kernel.h"

char *GetFile(FILE *f, int *length);
float *GetFloats(char *data, int data_length, char *seperator, int *floats);
float Medians(float *gpuNumber, size_t numbers, size_t outputSize);
float MedianOfMedians(float *gpuNumber, int numbers, int index);
int CompareFloats(const void *pa, const void *pb);
int WriteElements(float *output, float *input, int length, float median, int compareValue, int resultOffset);

#endif
