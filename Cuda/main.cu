
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
    ScoreGenome result = CudaGenetic(matrix, settings);
    timer.End();

    cout << "Done!" << endl;

    SaveData("cuda.txt", result);
    SaveTime("cuda.txt", timer.GetResult());

    cout << "Score: " << result.score << endl;
    //cin.get();

    return 0;
}

