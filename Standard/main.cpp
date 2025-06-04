#include <iostream>
#include <vector>
#include "Structs.h"
#include "Common.h"
#include "Standard.h"
#include "Timer.h"


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
        ScoreGenome result = StandardGenetic(matrix, settings);
        timer.End();

        int time = timer.GetResult().count();
        cout << sum << ", ";
        Times.push_back(timer.GetResult());

    }

    freeMatrix(matrix);

    SaveTimes("StandardTimes", Times);
    
    */
    
    string filename = "berlin52.txt";
    
    Settings settings;
    settings.iterations = 2000;
    Matrix matrix = loadData(filename);
    

    //printMatrix(matrix);
    
    Timer timer;
    timer.Start();
    ScoreGenome result = StandardGenetic(matrix, settings);
    timer.End();
    
    cout << "Done!" << endl;

    SaveData("standard.txt", result);
    SaveTime("standard.txt", timer.GetResult());

    //cin.get();

    return 0;
}
