#pragma once

struct Matrix
{
    float** m;
    int size;
};

struct Genome
{
    int* g;
    int size;
};

struct ScoreGenome
{
    float score;
    Genome genome;

};

struct Settings
{
    int iterations = 10000;
    int population = 68;
    int crossoversPerGenerations = 30;
    float mutationProp = 0.02f;
};