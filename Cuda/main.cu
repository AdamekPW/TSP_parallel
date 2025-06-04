
#include "Cuda.cuh"
#include <iostream>
#include "Common.h"
#include "Structs.h"
#include "Timer.h"

// crossy: 13.2
// typ mutacji (czy mutacja) 15.2989
// mut_tab * 3 

using namespace std;

int main()
{
    /*vector<chrono::microseconds> Times;

    Settings settings;
    Matrix matrix = loadData("berlin52.txt", "../Benchmarks/");
    for (int n = 100; n < 100 + 1000; n += 100)
    {
        settings.population = n;
        settings.crossoversPerGenerations = (int)(0.4 * n);
        int sum = settings.population + settings.crossoversPerGenerations * 2;


        Timer timer;
        timer.Start();
        ScoreGenome result = CudaGenetic(matrix, settings);
        timer.End();

        int time = timer.GetResult().count();
        cout << sum << ", ";
        Times.push_back(timer.GetResult());

    }

    freeMatrix(matrix);

    SaveTimes("CudaTimes", Times); */

    string filename = "berlin52.txt";
    Matrix matrix = loadData(filename);
    Settings settings;
    settings.iterations = 2000;

    Timer timer;
    timer.Start();
    ScoreGenome result = CudaGenetic(matrix, settings);
    timer.End();

    cout << "Done!" << endl;

    SaveData("cuda.txt", result);
    SaveTime("cuda.txt", timer.GetResult());

    cout << "Score: " << result.score << endl;
    //cin.get();

    return 0;
}

