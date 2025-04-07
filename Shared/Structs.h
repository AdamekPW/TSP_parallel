#pragma once

struct Matrix
{
    double** m;
    int size;
};

struct Genome
{
    int* g;
    int size;
};

struct ScoreGenome
{
    double score;
    Genome genome;
};

struct Settings
{
    int iterations = 20000;
    int population = 80;
    int crossoversPerGenerations = 30;
    float mutationProp = 0.02f;
};