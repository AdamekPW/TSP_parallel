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
    for (int n = 100; n < 100 + 1000; n += 100)
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

    }

    freeMatrix(matrix);
    SaveTimes("OpenMPTimes", Times);

    /*string filename = "bier127.txt";
    Matrix matrix = loadData(filename, "../Benchmarks/");

    Settings settings;

    Timer timer;

    timer.Start();
    ScoreGenome result = OpenMPGenetic(matrix, settings);
    timer.End();

    cout << "Done!" << endl;

    SaveData("openMP.txt", result);
    SaveTime("openMP.txt", timer.GetResult());

    cin.get();*/

    return 0;
}

