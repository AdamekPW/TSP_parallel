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
#include <set>
#include <utility> 

#include "Structs.h"

using namespace std;

#define MAX_N 1000

int RandomNumber(int LowerLimit, int UpperLimit);

Matrix RandomMatrix(int n);

float calcDistance(int x1, int y1, int x2, int y2);

void setDistance(Matrix& matrix, int from, int to, float distance);

Matrix loadData(string filename, string directory = "../Benchmarks/");

void SaveData(string filename, ScoreGenome& scoreGenome, string directory = "../Results/");

void SaveTime(string filename, chrono::microseconds time, string directory = "../Times/");

void SaveTimes(string filename, vector<chrono::microseconds>& times, string directory = "../Times/");

void freeMatrix(Matrix& matrix);

void freeGenome(Genome& genome);

void printMatrix(Matrix& matrix);

void printGenome(Genome& genome);