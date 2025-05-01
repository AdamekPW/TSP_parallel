#include "pch.h"
#include "Common.h"

int RandomNumber(int lowerLimit, int upperLimit)
{
    std::random_device rd;  // generator losowoœci (zwykle bazuj¹cy na sprzêcie)
    std::mt19937 gen(rd()); // silnik Mersenne Twister
    std::uniform_int_distribution<> distrib(lowerLimit, upperLimit - 1);
    return distrib(gen);
}

#include <set>
#include <utility> // std::pair

Matrix RandomMatrix(int n)
{
    Matrix matrix;

    int* coor_x = new int[n];
    int* coor_y = new int[n];
    std::set<std::pair<int, int>> used_points;

    int i = 0;
    while (i < n)
    {
        int x = RandomNumber(0, 10000);
        int y = RandomNumber(0, 10000);
        std::pair<int, int> point = std::make_pair(x, y);

        if (used_points.find(point) == used_points.end())
        {
            used_points.insert(point);
            coor_x[i] = x;
            coor_y[i] = y;
            i++;
        }
        // else: powtórzony punkt, wiêc losujemy ponownie
    }

    matrix.m = new float* [n];
    for (int i = 0; i < n; ++i)
        matrix.m[i] = new float[n];

    matrix.size = n;

    for (int from = 0; from < n; from++)
    {
        for (int to = 0; to < n; to++)
        {
            float distance = calcDistance(coor_x[from], coor_y[from], coor_x[to], coor_y[to]);
            setDistance(matrix, from, to, distance);
        }
    }

    delete[] coor_x;
    delete[] coor_y;

    return matrix;
}

float calcDistance(int x1, int y1, int x2, int y2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void setDistance(Matrix& matrix, int from, int to, float distance)
{
    matrix.m[from][to] = distance;
    matrix.m[to][from] = distance;
}

Matrix loadData(string filename, string directory)
{
    Matrix matrix;

    stringstream ss;
    ss << directory << filename;
    string filepath = ss.str();

    ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Opening file error: " << filename << std::endl;
    }

    int n, v;
    float x, y;

    file >> n;

    int* coor_x = new int[n];
    int* coor_y = new int[n];

    for (int i = 0; i < n; i++)
    {
        file >> v >> x >> y;
        coor_x[v - 1] = x;
        coor_y[v - 1] = y;
    }


    matrix.m = new float* [n];
    for (int i = 0; i < n; ++i)
        matrix.m[i] = new float[n];

    matrix.size = n;

    for (int from = 0; from < n; from++)
    {
        for (int to = 0; to < n; to++)
        {
            float distance = calcDistance(coor_x[from], coor_y[from], coor_x[to], coor_y[to]);
            setDistance(matrix, from, to, distance);
        }
    }

    delete[] coor_x;
    delete[] coor_y;

    return matrix;

}

void SaveData(string filename, ScoreGenome& scoreGenome, string directory)
{
    stringstream ss;
    ss << directory << filename;
    ofstream file(ss.str());
    if (!file.is_open())
    {
        std::cerr << "Opening file error: " << filename << std::endl;
        return;
    }

    file << scoreGenome.score << '\n';
    file << scoreGenome.genome.size << '\n';
    for (int i = 0; i < scoreGenome.genome.size; i++)
    {
        file << scoreGenome.genome.g[i] << '\n';
    }

    file.close();
}

void SaveTime(string filename, chrono::microseconds time, string directory)
{
    stringstream ss;
    ss << directory << filename;
    ofstream file(ss.str());
    if (!file.is_open())
    {
        std::cerr << "Opening file error: " << filename << std::endl;
        return;
    }

    file << time.count() << '\n';

    file.close();
}

void SaveTimes(string filename, vector<chrono::microseconds>& times, string directory)
{
    std::stringstream ss;
    ss << directory << filename;
    std::ofstream file(ss.str());

    if (!file.is_open())
    {
        std::cerr << "Opening file error: " << filename << std::endl;
        return;
    }

    for (const auto& time : times)
    {
        file << (float)time.count() / 1000000 << ", ";
    }

    file.close();
}

void freeMatrix(Matrix& matrix)
{
    for (int i = 0; i < matrix.size; i++)
        delete[] matrix.m[i];
    delete[] matrix.m;
}

void freeGenome(Genome& genome)
{
    delete[] genome.g;
}

void printMatrix(Matrix& matrix)
{
    for (int i = 0; i < matrix.size; i++)
    {
        for (int j = 0; j < matrix.size; j++)
        {
            cout << matrix.m[i][j] << " ";
        }
        cout << endl;
    }
}

void printGenome(Genome& genome)
{
    for (int i = 0; i < genome.size; i++)
    {
        cout << genome.g[i] << " ";
    }
    cout << endl;
}