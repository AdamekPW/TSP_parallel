#include <iostream>
#include <Structs.h>
#include <omp.h>
#include "Common.h"
#include "OpenMP.h"
#include "Timer.h"

using namespace std;

int main()
{

    vector<chrono::microseconds> Times;

    Settings settings;
    Matrix matrix = loadData("berlin52.txt", "../Benchmarks/");

    int baseSize = 100;
    for (int n = baseSize; n < 20 * baseSize; n += baseSize)
    {
        settings.population = n;
        settings.crossoversPerGenerations = (int)(0.4 * n);
        int sum = settings.population + settings.crossoversPerGenerations * 2;
    
        
        Timer timer;
        timer.Start();
        ScoreGenome result = OpenMPGenetic(matrix, settings);
        timer.End();

        int time = timer.GetResult().count();
        cout << sum << ", ";
        Times.push_back(timer.GetResult());

        delete[] result.genome.g;
    }

    freeMatrix(matrix);
    SaveTimes("OpenMPTimes", Times); 
    /*
    string filename = "berlin52.txt";
    Matrix matrix = loadData(filename, "../Benchmarks/");

    Settings settings;
    settings.iterations = 5000;

    Timer timer;

    timer.Start();
    ScoreGenome result = OpenMPGenetic(matrix, settings);
    timer.End();

    cout << "Done!" << endl;

    SaveData("openMP.txt", result);
    SaveTime("openMP.txt", timer.GetResult());

    */

    return 0;
}

