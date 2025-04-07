#include <iostream>
#include <Structs.h>
#include <omp.h>
#include "Common.h"
#include "OpenMP.h"
#include "Timer.h"

using namespace std;

int main()
{
    string filename = "berlin52.txt";
    Matrix matrix = loadData(filename, "../Benchmarks/");

    Settings settings;

    Timer timer;

    timer.Start();
    ScoreGenome result = OpenMPGenetic(matrix, settings);
    timer.End();

    cout << "Done!" << endl;

    SaveData("openMP.txt", result);
    SaveTime("openMP.txt", timer.GetResult());

    cin.get();

    return 0;
}

