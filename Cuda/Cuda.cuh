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

using namespace std;

struct CudaMatrix {
	float* m; 
	int size ;
};

struct CudaScoreGenome {

};

struct ScoreCompare {
	__host__ __device__
		bool operator()(const ScoreGenome& a, const ScoreGenome& b) const {
		return a.score < b.score;
	}
};

struct Params {
	int n;
	int population;
	int maxPopulation;
	int crossoversPerGeneration;
	float mutationProp;

	int randVecFrequency;
	int randVecSize;
	int randVecMul;
};


__device__ CudaMatrix d_cudaMatrix;
__device__ Params d_params;
__device__ ScoreGenome* d_scoreGenomes;

#define CROSSOVER_THREADS_PER_BLOCK 52

CudaMatrix ConvertToCudaMatrix(Matrix& matrix);
void FreeCudaMatrix(CudaMatrix& cudaMatrix);

int RandomNumber(int LowerLimit, int UpperLimit);
float Score(Matrix& matrix, Genome& genome);
void SimpleSample(Matrix& matrix, Genome& genome);


void GenerateRandomVec(int* vec, int size, int lowerLimit, int upperLimit);

__device__ int GetRandomInt(int lower, int upper, unsigned long seed, int threadId);
__global__ void RandomIntsKernel(int* output, int n, int lower, int upper, unsigned long seed);
__global__ void CrossoverGenomeCellKernel(int* randParentsVec, int* randCuttingPointsVec, int generation);
__device__ float CudaScore(Genome* genome);

__device__ void bitonic_sort(ScoreGenome* data, int size, ScoreCompare comp);
__global__ void sort_score_genomes(int size);

void AllocateAndCopyCudaMatrix(CudaMatrix& cudaMatrix);
void AllocateAndCopyScoreGenomes(ScoreGenome* scoreGenomes, int size);
void AllocateAndCopyParams(Params& params);
void FreeAndCopyScoreGenomes(ScoreGenome* scoreGenomes, int size);


ScoreGenome CudaGenetic(Matrix& matrix, Settings settings);