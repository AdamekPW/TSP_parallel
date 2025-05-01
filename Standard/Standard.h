#pragma once

#include "Structs.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <random>
#include "Common.h"

using namespace std;

float Score(Matrix& matrix, Genome& genome);
void SimpleSample(Matrix& matrix, Genome& genome);
bool IsInGenome(Genome& genome, int value, int endIndex);
void Crossover(Genome& g1_in, Genome& g2_in, Genome& g1_out, Genome& g2_out);
bool Mutate(Genome& genome, float propability);

ScoreGenome StandardGenetic(Matrix& matrix, Settings settings);

