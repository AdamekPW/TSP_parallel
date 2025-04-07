#include "Standard.h"




int RandomNumber(int lowerLimit, int upperLimit)
{
    std::random_device rd;  // generator losowo�ci (zwykle bazuj�cy na sprz�cie)
    std::mt19937 gen(rd()); // silnik Mersenne Twister
    std::uniform_int_distribution<> distrib(lowerLimit, upperLimit - 1);
    return distrib(gen);
}

double Score(Matrix& matrix, Genome& genome)
{
    double sum = 0;
    for (int i = 0; i < genome.size; i++)
    {
        int from = genome.g[i];
        int to = genome.g[(i + 1) % genome.size];
        sum += matrix.m[from][to];
    }
    return sum;
}

void SimpleSample(Matrix& matrix, Genome& genome)
{
    int n = matrix.size;
    int Start = RandomNumber(0, n);
    vector<bool> Visited(n, false);
    Visited[Start] = true;
    int VisitedCount = 1;

    int i = Start;
    while (VisitedCount <= n)
    {
        genome.g[VisitedCount - 1] = i;
        Visited[i] = true;

        double min1Value = numeric_limits<int>::max();
        int min1Index = -1;
        double min2Value = numeric_limits<int>::max();
        int min2Index = -1;

        for (int j = 0; j < n; j++)
        {
            if (matrix.m[i][j] > 0 && !Visited[j] && matrix.m[i][j] <= min1Value)
            {
                min2Value = min1Value;
                min2Index = min1Index;
                min1Value = matrix.m[i][j];
                min1Index = j;
            }
        }
        int Randomize = RandomNumber(0, 2);
        if (Randomize == 0 && min2Index != -1)
            i = min2Index;
        else
            i = min1Index;
        VisitedCount++;
    }
}

bool IsInGenome(Genome& genome, int value, int endIndex = -1)
{
    if (endIndex > genome.size || endIndex == -1)
        endIndex = genome.size;

    for (int i = 0; i < endIndex; i++)
    {
        if (genome.g[i] == value)
            return true;
    }
    return false;
}

bool Mutate(Genome& genome, float propability)
{
    int p = RandomNumber(0, 101);

    // Mutacja nie wystapi
    if (p > (int)(propability * 100))
        return false;

    p = RandomNumber(0, 101);
    if (p < 50) {
        /*cout << "Mutation1!" << endl;*/
        int index1 = RandomNumber(0, genome.size);
        int index2 = RandomNumber(0, genome.size);

        while (index1 == index2)
            index2 = RandomNumber(0, genome.size);

        int temp = genome.g[index1];
        genome.g[index1] = genome.g[index2];
        genome.g[index2] = temp;

    }
   else {
        int length = RandomNumber(2, (int)(genome.size / 2));

        int startIndex = RandomNumber(0, genome.size - length);
        int endIndex = startIndex + length;

        int copyStartPoint = RandomNumber(0, genome.size - length);
        int copyEndPoint = copyStartPoint + length;

        Genome newGenome;
        newGenome.size = genome.size;
        newGenome.g = new int[genome.size];

        for (int i = 0; i < newGenome.size; i++)
            newGenome.g[i] = -1;

        int index = copyStartPoint;
        for (int i = startIndex; i < endIndex; i++)
        {
            newGenome.g[index] = genome.g[i];
            index++;
        }

        index = 0;
        for (int i = 0; i < genome.size; i++)
        {
            while (index >= copyStartPoint && index < copyEndPoint) index++;

            if (!IsInGenome(newGenome, genome.g[i]) && index < newGenome.size)
            {
                newGenome.g[index] = genome.g[i];
                index++;
            }
        }

        delete[] genome.g;
        genome.g = newGenome.g;
        


    }
    return true;
}

void Crossover(Genome& g1_in, Genome& g2_in, Genome& g1_out, Genome& g2_out)
{
    int n = g1_in.size;
    int cuttingPoint = RandomNumber(1, n - 1);

    // Kopiuj lewe czesci od cuttingPoint do potomstwa
    for (int i = 0; i < cuttingPoint; i++)
    {
        g1_out.g[i] = g1_in.g[i];
        g2_out.g[i] = g2_in.g[i];
    }

    int g1_out_index = cuttingPoint;
    int g2_out_index = cuttingPoint;


    // Kopiuj kolejno pozostale, o ile nie wystepuja w potomstwie
    for (int i = 0; i < n; i++)
    {
        if (!IsInGenome(g1_out, g2_in.g[i], g1_out_index))
        {
            g1_out.g[g1_out_index] = g2_in.g[i];
            g1_out_index++;
        }

        if (!IsInGenome(g2_out, g1_in.g[i], g2_out_index))
        {
            g2_out.g[g2_out_index] = g1_in.g[i];
            g2_out_index++;
        }
    }


}

ScoreGenome StandardGenetic(Matrix& matrix, Settings settings)
{


    int maxPopulation = settings.population + settings.crossoversPerGenerations * 2;

    int n = matrix.size;
    ScoreGenome* scoreGenomes = new ScoreGenome[maxPopulation];

    for (int i = 0; i < maxPopulation; i++)
    {
        scoreGenomes[i].genome.g = new int[n];
        scoreGenomes[i].genome.size = n;
        SimpleSample(matrix, scoreGenomes[i].genome);
        scoreGenomes[i].score = Score(matrix, scoreGenomes[i].genome);
    }

    std::sort(scoreGenomes, scoreGenomes + settings.population, [](const ScoreGenome& a, const ScoreGenome& b) {
        return a.score < b.score;
        });

    double best = scoreGenomes[0].score;

    for (int generation = 0; generation < settings.iterations; generation++)
    {
        for (int c = 0; c < settings.crossoversPerGenerations; c++)
        {
            int parent1 = RandomNumber(0, settings.population);
            int parent2 = RandomNumber(0, settings.population);
            while (parent1 == parent2) parent2 = RandomNumber(0, settings.population);

            int index1 = settings.population + c * 2;
            int index2 = settings.population + c * 2 + 1;


            Crossover(scoreGenomes[parent1].genome, scoreGenomes[parent2].genome,
                scoreGenomes[index1].genome, scoreGenomes[index2].genome);

            scoreGenomes[index1].score = Score(matrix, scoreGenomes[index1].genome);
            scoreGenomes[index2].score = Score(matrix, scoreGenomes[index2].genome);

        }

        for (int m = 0; m < maxPopulation; m++)
        {
            if (Mutate(scoreGenomes[m].genome, settings.mutationProp))
            {
                scoreGenomes[m].score = Score(matrix, scoreGenomes[m].genome);
            }

        }

        std::sort(scoreGenomes, scoreGenomes + maxPopulation, [](const ScoreGenome& a, const ScoreGenome& b) {
            return a.score < b.score;
            });

        if (scoreGenomes[0].score < best)
        {
            best = scoreGenomes[0].score;
            cout << best << endl;

        }

    }


    ScoreGenome result;
    result.score = scoreGenomes[0].score;
    result.genome.size = n;
    result.genome.g = new int[n];

    for (int i = 0; i < n; i++)
    {
        result.genome.g[i] = scoreGenomes[0].genome.g[i];
    }

    for (int i = 0; i < maxPopulation; i++)
    {
        delete[] scoreGenomes[i].genome.g;
    }

    delete[] scoreGenomes;

    return result;

}

