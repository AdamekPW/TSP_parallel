#include <iostream>
#include "Structs.h"
#include "Common.h"
#include "Standard.h"
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

    SaveData("standard.txt", result);
    SaveTime("standard.txt", timer.GetResult());

    cin.get();

    return 0;
}
