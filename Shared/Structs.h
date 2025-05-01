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
    int iterations = 2000;
    int population = 160;
    int crossoversPerGenerations = 48;
    float mutationProp = 0.02f;
};