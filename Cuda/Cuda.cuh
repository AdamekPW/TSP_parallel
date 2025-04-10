#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "Structs.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <random>

using namespace std;

struct CudaMatrix {
	float* m; 
	int size;
};


int RandomNumber(int LowerLimit, int UpperLimit);

float Score(Matrix& matrix, Genome& genome);
void SimpleSample(Matrix& matrix, Genome& genome);
bool IsInGenome(Genome& genome, int value, int endIndex);
void Crossover(Genome& g1_in, Genome& g2_in, Genome& g1_out, Genome& g2_out);
bool Mutate(Genome& genome, float propability);

ScoreGenome StandardGenetic(Matrix& matrix, Settings settings);
