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
	int* genome;
	float score;
};

struct ScoreCompare {
	__host__ __device__
		bool operator()(const CudaScoreGenome& a, const CudaScoreGenome& b) const {
		return a.score < b.score;
	}
};

struct Params {
	int n;
	int iterations;
	int population;
	int maxPopulation;
	int crossoversPerGeneration;
	float mutationProp;

};

#define CROSSOVER_THREADS_PER_BLOCK 52

int RandomNumber(int lowerLimit, int upperLimit);
float Score(Matrix& matrix, Genome& genome);
void SimpleSample(Matrix& matrix, Genome& genome);
ScoreGenome* FullSimpleSample(Matrix& matrix, Params& params);

__global__ void GenerateRandomIntsKernel(int* d_array, int size, int lowerLimit, int upperLimit, unsigned long seed);
__device__ float CudaScore(int* d_genome);
__global__ void CrossoverKernel(int* d_mutationRandTable, int iteration);
__global__ void MutationKernel(int* d_mutationRandTable, int* d_randT1, int* d_randT2, int* d_randT3, int iteration);

void GenerateRandomIntsOnGPU(int* d_array, int size, int lowerLimit, int upperLimit, int seedOffset);
void AllocateCudaMatrix(int size);
void FreeCudaMatrix();
void CopyCudaMatrixFromHostToDevice(Matrix& matrix);

void CopyParamsFromHostToDevice(const Params &h_params);

void AllocateCudaScoreGenomes(int maxPopulation, int n);
void FreeCudaScoreGenomes(int maxPopulation);
void CopyCudaScoreGenomesFromHostToDevice(ScoreGenome* h_scoreGenomes, int maxPopulation);
ScoreGenome CopyScoreGenomeFromDeviceToHost(int index, int n);

void AllocateRandTables(Params& params, int** d_crossoversRandTable, int** d_mutationRandTable);
void RandTablesInit(Params& params, int* d_crossoversRandTable, int* d_mutationRandTable);
void FreeRandTables(int* d_crossoversRandTable, int* d_mutationRandTable);

ScoreGenome CudaGenetic(Matrix& matrix, Settings settings);


