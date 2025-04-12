#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <curand.h>

#include "Structs.h"
#include <stdio.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <random>

using namespace std;

struct CudaMatrix {
	float* m = nullptr; 
	int size = 0;
};

struct RandVec {
	int* vec = nullptr;
	int frequency = 1500;
	int size = 0;
	int mul = 0; // potrzebne przy indeksowaniu w generacjach
};


CudaMatrix ConvertToCudaMatrix(Matrix& matrix);
void FreeCudaMatrix(CudaMatrix& cudaMatrix);

int RandomNumber(int LowerLimit, int UpperLimit);
void GenerateRandomVec(int* vec, int size, int lowerLimit, int upperLimit);


float Score(Matrix& matrix, Genome& genome);
__device__ float CudaScore(CudaMatrix* cudaMatrix, Genome* genome);
void SimpleSample(Matrix& matrix, Genome& genome);
bool IsInGenome(Genome& genome, int value, int endIndex);
__device__  bool CudaIsInGenome(Genome* genome, int value, int endIndex);
void Crossover(Genome& g1_in, Genome& g2_in, Genome& g1_out, Genome& g2_out);
__device__  void CudaCrossover(Genome* g1_in, Genome* g2_in, Genome* g1_out, Genome* g2_out, int cuttingPoint);
bool Mutate(Genome& genome, float propability);
ScoreGenome StandardGenetic(Matrix& matrix, Settings settings);


CudaMatrix* AllocateAndCopyCudaMatrixFromCPU(CudaMatrix& cudaMatrix);
ScoreGenome* AllocateAndCopyScoreGenomesFromCPU(ScoreGenome* scoreGenomes, int size);
void CopyAndFreeScoreGenomesFromGPU(ScoreGenome* d_scoreGenomes, int size, ScoreGenome* scoreGenomes);
ScoreGenome CudaGenetic(Matrix& matrix, Settings settings);