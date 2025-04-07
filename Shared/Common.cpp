#include "pch.h"
#include "Common.h"

double calcDistance(int x1, int y1, int x2, int y2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void setDistance(Matrix& matrix, int from, int to, double distance)
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
    double x, y;

    file >> n;

    int* coor_x = new int[n];
    int* coor_y = new int[n];

    for (int i = 0; i < n; i++)
    {
        file >> v >> x >> y;
        coor_x[v - 1] = x;
        coor_y[v - 1] = y;
    }


    matrix.m = new double* [n];
    for (int i = 0; i < n; ++i)
        matrix.m[i] = new double[n];

    matrix.size = n;

    for (int from = 0; from < n; from++)
    {
        for (int to = 0; to < n; to++)
        {
            double distance = calcDistance(coor_x[from], coor_y[from], coor_x[to], coor_y[to]);
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