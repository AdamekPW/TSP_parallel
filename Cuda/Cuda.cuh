#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "Structs.h"
#include "Common.h"
#include <stdio.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <random>
#include <omp.h>

using namespace std;

struct CudaMatrix {
	float* m; 
	int size ;
};

struct CudaScoreGenome {
	int* genome;
	float score;
};

struct ScoreCompare {
	__host__ __device__
		bool operator()(const CudaScoreGenome& a, const CudaScoreGenome& b) const {
		return a.score < b.score;
	}
};



#define MAX_THREADS 64
#define N 52

float Score(Matrix& matrix, Genome& genome);

void AllocateCudaMatrix(int size);
void FreeCudaMatrix();
void CopyCudaMatrixFromHostToDevice(Matrix& matrix);

ScoreGenome CudaGenetic(Matrix& matrix, Settings settings);

void Crossover(int* g1_in, int* g2_in, int* g1_out, int* g2_out, int n);

__device__ float GetScore(int* sequence, int n);
__device__ bool IsConsistent(int* sequence, int n);
__device__ int DeviceRandomNumber(int lowerLimit, int upperLimit, int idx, unsigned int seed);
__device__ int DeviceRandomNumber(int lowerLimit, int upperLimit, curandState* state);

__global__ void InitPopulationKernel(int* d_population, int maxPopulation, int n, curandState* statesTable);
__global__ void InitRandomStatesKernel(curandState* states, unsigned int seed);
__global__ void MutationKernel(int* population, int n, curandState* d_curandStates);
