#pragma once

#include <iostream>
#include <algorithm>
#include <stack>
#include <limits>
#include <string>
#include <sstream>
#include <set>
#include <random>
#include <unordered_set>
#include <fstream>
#include <ctime>
#include <chrono>
#include "Structs.h"

using namespace std;

float calcDistance(int x1, int y1, int x2, int y2);

void setDistance(Matrix& matrix, int from, int to, float distance);

Matrix loadData(string filename, string directory = "../Benchmarks/");

void SaveData(string filename, ScoreGenome& scoreGenome, string directory = "../Results/");

void SaveTime(string filename, chrono::microseconds time, string directory = "../Times/");

void freeMatrix(Matrix& matrix);

void freeGenome(Genome& genome);

void printMatrix(Matrix& matrix);

void printGenome(Genome& genome);