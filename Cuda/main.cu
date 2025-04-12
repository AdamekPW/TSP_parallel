
#include "Cuda.cuh"
#include <iostream>
#include "Common.h"
#include "Structs.h"
#include "Timer.h"



using namespace std;

int main()
{
   /*Timer timer;
    int n = 200000;
    int* vec = new int[n];

    timer.Start();
    GenerateRandomVec(vec, n, 0, 20);
    timer.End();
    long R = (long)timer.GetResult().count();
    cout << "Wersja rownolegla: " << R << endl;

    timer.Start();
    for (int i = 0; i < n; i++)
    {
        vec[i] = RandomNumber(0, 20);
    }
    timer.End();
    long S = (long)timer.GetResult().count();
    cout << "Wersja szeregowa:" << S << endl;

    float speed = (float)R / (float)S;
    cout << speed << endl;

    delete[] vec;*/
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

    cin.get();

    return 0;
}

