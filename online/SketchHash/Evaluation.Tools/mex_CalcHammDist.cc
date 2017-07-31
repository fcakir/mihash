#include "mex.h"

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <stdint.h>

const int kCompHashCodeCap = 64; // the capacity of each compact hash code bit
const double kEpsilon = 0.0000000001;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
// input paramters:
// 1. hashCode_1
// 2. hashCode_2
// output parameters:
// 1. distMat
{
    if (nrhs != 2)
    {
        mexErrMsgTxt("invalid input parameters\n");
    }
    
    // assign pointer for <hashCode_1>
    int cntRowHashCode_1 = mxGetM(prhs[0]);
    int cntColHashCode_1 = mxGetN(prhs[0]);
    uint8_t* pHashCode_1 = (uint8_t*)mxGetData(prhs[0]);
    
    // assign pointer for <hashCode_2>
    int cntRowHashCode_2 = mxGetM(prhs[1]);
    int cntColHashCode_2 = mxGetN(prhs[1]);
    uint8_t* pHashCode_2 = (uint8_t*)mxGetData(prhs[1]);

    // check if the hash code length is the same
    if (cntRowHashCode_1 != cntRowHashCode_2)
    {
        mexErrMsgTxt("input data's hash code length must be the same\n");
    }
    
    // obtain basic variables' values
    int hashCodeLen = cntRowHashCode_1;
    int instCnt_1 = cntColHashCode_1;
    int instCnt_2 = cntColHashCode_2;

    // allocate space for compact hash code
    int compHashCodeLen = (hashCodeLen - 1) / kCompHashCodeCap + 1;
    uint64_t* pCompHashCode_1 = (uint64_t*)mxMalloc(sizeof(uint64_t) * compHashCodeLen * instCnt_1);
    uint64_t* pCompHashCode_2 = (uint64_t*)mxMalloc(sizeof(uint64_t) * compHashCodeLen * instCnt_2);
    
    // assign pointer for <distMat>
    int cntRowDistMat = instCnt_1;
    int cntColDistMat = instCnt_2;
    plhs[0] = mxCreateNumericMatrix(cntRowDistMat, cntColDistMat, mxUINT16_CLASS, mxREAL);
    uint16_t* pDistMat = (uint16_t*)mxGetPr(plhs[0]);
    #define distMat(r, c) pDistMat[(c) * cntRowDistMat + (r)]
    
    #pragma omp parallel for num_threads(8)

    // convert original hash code into compact hash code
    for (int instInd_1 = 0; instInd_1 < instCnt_1; instInd_1++)
    {
        for (int compHashCodeInd = 0; compHashCodeInd < compHashCodeLen; compHashCodeInd++)
        {
            uint64_t compHashBitVal = 0;
            int hashCodeIndBeg = kCompHashCodeCap * compHashCodeInd;
            int hashCodeIndEnd = std::min(hashCodeLen, hashCodeIndBeg + kCompHashCodeCap);
            int loopCnt = hashCodeIndEnd - hashCodeIndBeg;
            const uint8_t* hashCodePtr = pHashCode_1 + (hashCodeLen * instInd_1 + hashCodeIndBeg);

            for (int loopInd = loopCnt; loopInd > 0; loopInd--)
            {
                compHashBitVal = (compHashBitVal << 1) | *(hashCodePtr++);
            }
            *(pCompHashCode_1 + (compHashCodeLen * instInd_1 + compHashCodeInd)) = compHashBitVal;
        }
    }
    for (int instInd_2 = 0; instInd_2 < instCnt_2; instInd_2++)
    {
        for (int compHashCodeInd = 0; compHashCodeInd < compHashCodeLen; compHashCodeInd++)
        {
            uint64_t compHashBitVal = 0;
            int hashCodeIndBeg = kCompHashCodeCap * compHashCodeInd;
            int hashCodeIndEnd = std::min(hashCodeLen, hashCodeIndBeg + kCompHashCodeCap);
            int loopCnt = hashCodeIndEnd - hashCodeIndBeg;
            const uint8_t* hashCodePtr = pHashCode_2 + (hashCodeLen * instInd_2 + hashCodeIndBeg);
            
            for (int loopInd = loopCnt; loopInd > 0; loopInd--)
            {
                compHashBitVal = (compHashBitVal << 1) | *(hashCodePtr++);
            }
            *(pCompHashCode_2 + (compHashCodeLen * instInd_2 + compHashCodeInd)) = compHashBitVal;
        }
    }

    // compute pairwise Hamming distance
    for (int instInd_1 = 0; instInd_1 < instCnt_1; instInd_1++)
    {
        for (int instInd_2 = 0; instInd_2 < instCnt_2; instInd_2++)
        {
            uint16_t distVal = 0;
            const uint64_t* compHashCodePtr_1 = pCompHashCode_1 + (compHashCodeLen * instInd_1);
            const uint64_t* compHashCodePtr_2 = pCompHashCode_2 + (compHashCodeLen * instInd_2);

            for (int compHashCodeInd = 0; compHashCodeInd < compHashCodeLen; compHashCodeInd++)
            {
                distVal += __builtin_popcountll(compHashCodePtr_1[compHashCodeInd] ^ compHashCodePtr_2[compHashCodeInd]);
            }

            distMat(instInd_1, instInd_2) = distVal;
        }
    }
}
