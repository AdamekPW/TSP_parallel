
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
    /*
    int size = 600000;
    int lowerLimit = 0;
    int upperLimit = 100;
    Timer timer;
    timer.Start();
    
    for (int i = 0; i < size; i++)
    {
 
        int t = RandomNumber(lowerLimit, upperLimit);
    }

    timer.End();
    long s = timer.GetResult().count();
    cout << "Sekwencyjnie: " << s << endl;

    
    int* d_array;
    cudaMalloc(&d_array, size * sizeof(int));
    
    timer.Start();
    GenerateRandomIntsOnGPU(d_array, size, lowerLimit, upperLimit);
    timer.End();

    long r = timer.GetResult().count();
    cout << "Rownolegle: " << r << endl;

    cout << "Boost: " << (float)s / (float)r << endl;

    cout << "Zuzywana pamiec GPU: " << (float)(sizeof(int) * size) / (1024*1024) << " MB" << endl;

    // Skopiowanie danych do hosta i wypisanie przykładów
    int* h_array = new int[size];
    cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    //for (int i = 0; i < 10; ++i)
    //    std::cout << h_array[i] << " ";
    //std::cout << std::endl;
    delete[] h_array;
    cudaFree(d_array);
    */

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

