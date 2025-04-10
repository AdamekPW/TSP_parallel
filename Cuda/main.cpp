
#include "Cuda.cuh"
#include <iostream>
#include "Common.h"
#include "Structs.h"
#include "Timer.h"



using namespace std;

int main()
{
    string filename = "berlin52.txt";
    Matrix matrix = loadData(filename);

    Settings settings;

    Timer timer;
    timer.Start();
    ScoreGenome result = StandardGenetic(matrix, settings);
    timer.End();

    cout << "Done!" << endl;

    SaveData("cuda.txt", result);
    SaveTime("cuda.txt", timer.GetResult());

    cin.get();

    return 0;
}

