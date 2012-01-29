#include "main.h"

/* return the raw data of a file, while trying not to fragment memory too much... */
char *GetFile(FILE *f, int *length) {
    char *ret;
    int l;
    assert(!fseek(f, 0, SEEK_END));
    l = ftell(f);
    assert(!fseek(f, 0, SEEK_SET));
    if (length)
        *length = l;
    assert(ret = (char *) malloc(l));
    assert(fread(ret, 1, l, f) == l);
    return ret;
}

/* convert a series of floats, represented in ASCII in /data/, seperated by /seperator/ into an array of floats (which is returned).  Store the length in /floats/, if it isn't NULL. */
float *GetFloats(char *data, int data_length, char *seperator, int *floats) {
    float *ret;
    assert(ret = (float *) malloc(1024*sizeof(float)));
    int index = 0;
    int cur_size = 1024;
    char *endptr;
    int len_seperator = strlen(seperator);
    while (data_length >= 0) {
        ret[index] = strtof(data, &endptr);
        if (data != endptr) {
            index++;
            if (index >= cur_size) {
                cur_size *= 2;
                assert(ret = (float *) realloc(ret, cur_size * sizeof(float)));
            }
        }
        if (*endptr)
            assert(!memcmp(endptr, seperator, len_seperator));
        else
            break;
        data_length = (endptr + len_seperator) - data;
        data = endptr + len_seperator;
    }
    if (floats)
        *floats = index;
    return ret;
}

/* compare two floats, in the manner that libc's qsort wants */
int CompareFloats(const void *pa, const void *pb) {
    float a = *((float *)pa);
    float b = *((float *)pb);
    if (a < b)
        return -1;
    else if (a == b) // technically I should consider an epsilon comparison, but I doubt it'll matter here
        return 0;
    else
        return 1;
}

/* get the median of medians.  P.S. Make sure to pass the numbers in as a gpu pointer, as this function is recursive. */
float Medians(float *gpuNumber, size_t numbers) {
    float *gpuOutput;
    float *cpuOutput;
    int blocks = (numbers + BLOCK_SIZE-1) / BLOCK_SIZE;
    int arraySize = blocks*RESULT_SIZE;
    int realArraySize = arraySize;
    // make real array size a multiple of BLOCK_SIZE
    if (realArraySize % BLOCK_SIZE != 0) {
        realArraySize = arraySize + BLOCK_SIZE - (arraySize % BLOCK_SIZE);
        CUDA_SAFE_CALL(cudaMalloc((void **)&gpuOutput, realArraySize * sizeof(float)));
        // now pad the last few numbers with inf, because that's what the kernel expects, as it doesn't muck with the algorithm
        float finf = INFINITY;
        int iinf = *((int *)&finf); // get the bits of the float
        CUDA_SAFE_CALL(cudaMemset(gpuOutput + arraySize, iinf, (BLOCK_SIZE - (arraySize % BLOCK_SIZE))*sizeof(float)));
    } else {
        // if the user was nice to us, we don't have to do anything
        CUDA_SAFE_CALL(cudaMalloc((void **)&gpuOutput, arraySize * sizeof(float)));
    }
    dim3 numThreads(BLOCK_SIZE, 1);
    dim3 numBlocks(blocks, 1);
    FindMedians<<<numBlocks,numThreads>>>(gpuNumber, gpuOutput);
    cudaThreadSynchronize();
    if (blocks*RESULT_SIZE >= BLOCK_SIZE) {
        float median =  Medians(gpuOutput, realArraySize);
        CUDA_SAFE_CALL(cudaFree(gpuOutput));
        return median;
    }
    /* this could be run on the GPU, but the speed difference is negligable, as the array size is always < 256 elements */
    assert(cpuOutput = (float *)malloc(blocks*RESULT_SIZE*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(cpuOutput, gpuOutput, arraySize*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(gpuOutput));
    qsort(cpuOutput, arraySize, sizeof(float), CompareFloats);
    return cpuOutput[arraySize/2 - 1];
}

void PrintGpuInts(char *name, int *gpuNumbers, int numbers) {
    int *cpuNumbers, i;
    assert(cpuNumbers = (int *) malloc(numbers*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(cpuNumbers, gpuNumbers, numbers*sizeof(int), cudaMemcpyDeviceToHost));
    for (i = 0; i < numbers; i++) {
        printf("%s[%i] = %i\n", name, i, cpuNumbers[i]);
    }
}

void PrintGpuFloats(char *name, float *gpuNumbers, int numbers) {
    float *cpuNumbers;
    int i;
    assert(cpuNumbers = (float *) malloc(numbers*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(cpuNumbers, gpuNumbers, numbers*sizeof(float), cudaMemcpyDeviceToHost));
    for (i = 0; i < numbers; i++) {
        printf("%s[%i] = %f\n", name, i, cpuNumbers[i]);
    }
}


int WriteElements(float *output, float *input, int length, float median, int compareValue, int resultOffset) {
    int *compareResult;
    CUDA_SAFE_CALL(cudaMalloc((void **) &compareResult, length*sizeof(int)));
    int blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 positionThreads(BLOCK_SIZE, 1);
    dim3 positionBlocks(blocks, 1);
    CalcRelativePosition<<<positionBlocks, positionThreads>>>(input, compareResult, median, compareValue);
    cudaThreadSynchronize();
    dim3 prefixThreads(BLOCK_SIZE/2, 1);
    dim3 prefixBlocks(1, 1);
    int i, *prefixResult, *prefixOffset;
    CUDA_SAFE_CALL(cudaMalloc((void **) &prefixResult, length*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &prefixOffset, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(prefixOffset, &resultOffset, sizeof(int), cudaMemcpyHostToDevice));
    for (i = 0; i < length; i += 256) {
        PrefixScan<<<prefixBlocks, prefixThreads>>>(&compareResult[i], &prefixResult[i], prefixOffset);
        cudaThreadSynchronize();
    }
    PrintGpuInts("prefixResult", prefixResult, length);
    ReorderElements<<<positionBlocks, positionThreads>>>(input, output, prefixResult, median, compareValue);
    cudaThreadSynchronize();
    PrintGpuFloats("newNumbers", output, length);
    int returnValue;
    CUDA_SAFE_CALL(cudaMemcpy(&returnValue, prefixOffset, sizeof(int), cudaMemcpyDeviceToHost));
    return returnValue;
}

/* perform the full median of medians algorithm.  Again, keep in mind that
 * these numbers must be stored on the GPU, not the CPU */
 float *infinities;

float CpuMedianOfMedians(float *gpuNumber, int numbers, int index) {
    float *cpuNumber;
    assert(cpuNumber = (float *) malloc(numbers*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(cpuNumber, gpuNumber, numbers*sizeof(float), cudaMemcpyDeviceToHost));
    qsort(cpuNumber, numbers, sizeof(float), CompareFloats);
    return cpuNumber[index];
}

float MedianOfMedians(float *gpuNumber, int numbers, int index) {
    if (numbers <= 256)
        return CpuMedianOfMedians(gpuNumber, numbers, index);
    float median = Medians(gpuNumber, numbers);
    printf("Median of Medians: %f\n", median);
    float *output;
    CUDA_SAFE_CALL(cudaMalloc((void **) &output, (numbers+255)*sizeof(int)));
    int lt_offset = WriteElements(output, gpuNumber, numbers, median, LESS_THAN, 0);
    int eq_offset = WriteElements(output, gpuNumber, numbers, median, EQUAL_TO, lt_offset);
    int gt_offset = WriteElements(output, gpuNumber, numbers, median, GREATER_THAN, eq_offset);
    if (index >= lt_offset && index < eq_offset)
        return median;
    else if (index < lt_offset) {
        if (lt_offset % 256 != 0 && lt_offset > 256) { /* not a multiple of 256 */
            CUDA_SAFE_CALL(cudaMemcpy(output, infinities, (256 - (lt_offset % 256))*sizeof(float), cudaMemcpyHostToDevice));
            lt_offset += 256 - (lt_offset % 256);
        }
        return MedianOfMedians(output, lt_offset, index);
    } else { /* index > eq_offset */
        int length = numbers - eq_offset;
        int start = eq_offset;
        if (length % 256 != 0 && length > 256) {
            CUDA_SAFE_CALL(cudaMemcpy(&output[start], infinities, (256 - (length % 256))*sizeof(float), cudaMemcpyHostToDevice));
            length += 256 - (length % 256);
        }
        return MedianOfMedians(&output[start], length, index - start);
    }
}

void InitializeInfinities() {
    assert(infinities = (float *) malloc(BLOCK_SIZE*sizeof(float)));
    int i;
    for (i = 0; i < BLOCK_SIZE; i++) {
        infinities[i] = INFINITY;
    }
}

int main(int argc, char **argv) {
    FILE *help = NULL;
    FILE *numbers = NULL;
    char *error_message = NULL;
    char *seperator = "\n";
    long long element = -1;
    int i;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            help = stdout;
            break;
        } else if (!strcmp(argv[i], "--filename") || !strcmp(argv[i], "-f")) {
            if ((i + 1) >= argc) {
                error_message = "No filename specified";
                help = stderr;
                break;
            }
            ++i;
            numbers = fopen(argv[i], "r");
            if (!numbers) {
                assert(error_message = (char *) malloc(strlen(argv[i]) + 1024));
                sprintf(error_message, "Unable to open %s.  Ensure the file exists and you have read permissions.", argv[i]);
                help = stderr;
                break;
            }
        } else if (!strcmp(argv[i], "-e") || !strcmp(argv[i], "--element")) {
            if ((i + 1) >= argc) {
                error_message = "No number specified, please try again.";
                help = stderr;
                break;
            }
            i++;
            element = strtoll(argv[i], &error_message, 10);
            if (*error_message) { /* not a number */ 
                error_message = "The element specified was not an integer. Please specify an integer (in base 10).";
                help = stderr;
                break;
            }
            if (element < 0) {
                error_message = "Cannot select a negatively indexed element, as they do not exist.  Please try again.";
                help = stderr;
                break;
            }
            error_message = NULL;
        } else if (!strcmp(argv[i], "--seperator") || !strcmp(argv[i], "-s")) {
            if ((i + 1) >= argc) {
                error_message = "No seperator specified.  Aborting.";
                help = stderr;
                break;
            }
            i++;
            seperator = strdup(argv[i]);
        } else {
            assert(error_message = (char *)malloc(strlen(argv[i]) + 1024));
            sprintf(error_message, "Unknown parameter: %s", argv[i]);
            help = stderr;
            break;
        }
    }
    if (!help) { /* check to ensure they /really/ don't need help */
        if (element == -1) {
            error_message = "No element set";
            help = stderr;
        }
        if (numbers == NULL) {
            error_message = "No file specified";
            help = stderr;
        }
    }
    if (help) {
        if (error_message)
            fprintf(help, "%s\n", error_message);
        fprintf(help, "Options for %s\n", argv[0]);
        fprintf(help, "-f|--filename name\n");
        fprintf(help, "-e|--element e\n");
        fprintf(help, "-s|--seperator str (defaults to \\n)\n");
        fprintf(help, "-h|--help\n");
        if (help == stdout)
            return 0;
        return 1;
    }
    InitializeInfinities();
    int file_length;
    char *file_data = GetFile(numbers, &file_length);
    int num_floats;
    float *floats = GetFloats(file_data, file_length, seperator, &num_floats);
    float *gpuFloats;
    CUDA_SAFE_CALL(cudaMalloc((void **) &gpuFloats, num_floats*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(gpuFloats, floats, num_floats*sizeof(float), cudaMemcpyHostToDevice));
    printf("The final result is: %f\n", MedianOfMedians(gpuFloats, num_floats, element));
    CUDA_SAFE_CALL(cudaFree(gpuFloats));
}
