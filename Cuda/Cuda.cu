#include "Cuda.cuh"


CudaMatrix ConvertToCudaMatrix(Matrix& matrix)
{
    int n = matrix.size;
    CudaMatrix cudaMatrix;
    cudaMatrix.size = n * n;
    cudaMatrix.m = new float[cudaMatrix.size];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cudaMatrix.m[i * n + j] = matrix.m[i][j];
        }
    }

    return cudaMatrix;
}

void FreeCudaMatrix(CudaMatrix& cudaMatrix)
{
    delete[] cudaMatrix.m;
}

int RandomNumber(int lowerLimit, int upperLimit)
{
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distrib(lowerLimit, upperLimit - 1);
    return distrib(gen);
}

__global__ void generateRandomInts(int* output, int n, int lower, int upper, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState state;
    curand_init(seed, idx, 0, &state); 

    float rand_uniform = curand_uniform(&state);
    int range = upper - lower;
    int randInt = lower + static_cast<int>(rand_uniform * range);

    
    if (randInt >= upper) randInt = upper - 1;

    output[idx] = randInt;
}

void GenerateRandomVec(int* d_vec, int size, int lowerLimit, int upperLimit)
{
    generateRandomInts << <(size + 127) / 128, 128 >> > (d_vec, size, lowerLimit, upperLimit, time(NULL));
}

float Score(Matrix& matrix, Genome& genome)
{
    float sum = 0;
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

        float min1Value = numeric_limits<int>::max();
        int min1Index = -1;
        float min2Value = numeric_limits<int>::max();
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

__device__ void CudaCrossover(Genome* g1_in, Genome* g2_in, Genome* g1_out, Genome* g2_out, int cuttingPoint)
{
    int n = g1_in->size;

    // Kopiuj lewe czesci od cuttingPoint do potomstwa
    for (int i = 0; i < cuttingPoint; i++)
    {
        g1_out->g[i] = g1_in->g[i];
        g2_out->g[i] = g2_in->g[i];
    }

    int g1_out_index = cuttingPoint;
    int g2_out_index = cuttingPoint;


    // Kopiuj kolejno pozostale, o ile nie wystepuja w potomstwie
    for (int i = 0; i < n; i++)
    {
        if (!CudaIsInGenome(g1_out, g2_in->g[i], g1_out_index))
        {
            g1_out->g[g1_out_index] = g2_in->g[i];
            g1_out_index++;
        }

        if (!CudaIsInGenome(g2_out, g1_in->g[i], g2_out_index))
        {
            g2_out->g[g2_out_index] = g1_in->g[i];
            g2_out_index++;
        }
    }

}

__device__ bool CudaIsInGenome(Genome* genome, int value, int endIndex = -1)
{
    if (endIndex > genome->size || endIndex == -1)
        endIndex = genome->size;

    for (int i = 0; i < endIndex; i++)
    {
        if (genome->g[i] == value)
            return true;
    }
    return false;
}

__device__ float CudaScore(CudaMatrix* cudaMatrix, Genome* genome)
{
    float sum = 0;
    for (int i = 0; i < genome->size; i++)
    {
        int from = genome->g[i];
        int to = genome->g[(i + 1) % genome->size];
        sum += cudaMatrix->m[from * genome->size + to];
    }
    return sum;
}

__global__ void CrossoverKernel(ScoreGenome* scoreGenomes, CudaMatrix* cudaMatrix, 
    RandVec* randVec, Settings* settings, int generation) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    int n = scoreGenomes[0].genome.size;

    int parent1 = randVec->vec[(generation * randVec->mul) % randVec->frequency + 2 * c];
    int parent2 = randVec->vec[(generation * randVec->mul) % randVec->frequency + 2 * c + 1];
    if (parent1 == parent2)
    {
        int index = 0;
        while (parent1 == parent2 && index < randVec->size)
        {
            parent2 = randVec->vec[index];
            index++;
        }
    }

    int index1 = settings->population + c * 2;
    int index2 = settings->population + c * 2 + 1;

    int cuttingPoint = randVec->vec[(randVec->vec[index1] * randVec->vec[index2]) % randVec->size];
    if (cuttingPoint == 0) cuttingPoint = 1;
    else if (cuttingPoint == n - 1) cuttingPoint = n - 2;

    CudaCrossover(&scoreGenomes[parent1].genome, &scoreGenomes[parent2].genome,
        &scoreGenomes[index1].genome, &scoreGenomes[index2].genome, cuttingPoint);

    scoreGenomes[index1].score = CudaScore(cudaMatrix, &(scoreGenomes[index1].genome));
    scoreGenomes[index2].score = CudaScore(cudaMatrix, &(scoreGenomes[index2].genome));
}

void CrossoverLoop(ScoreGenome* scoreGenomes, Matrix &matrix, RandVec &randVec, Settings &settings, int generation) 
{
    for (int c = 0; c < settings.crossoversPerGenerations; c++)
    {   
        int parent1 = randVec.vec[(generation * randVec.mul) % randVec.frequency + 2 * c];
        int parent2 = randVec.vec[(generation * randVec.mul) % randVec.frequency + 2 * c + 1];
        if (parent1 == parent2)
        {
            int index = 0;
            while (parent1 == parent2 && index < randVec.size)
            {
                parent2 = randVec.vec[index];
                index++;
            }
        }

        int index1 = settings.population + c * 2;
        int index2 = settings.population + c * 2 + 1;


        Crossover(scoreGenomes[parent1].genome, scoreGenomes[parent2].genome,
            scoreGenomes[index1].genome, scoreGenomes[index2].genome);

        scoreGenomes[index1].score = Score(matrix, scoreGenomes[index1].genome);
        scoreGenomes[index2].score = Score(matrix, scoreGenomes[index2].genome);

    }
}

ScoreGenome StandardGenetic(Matrix& matrix, Settings settings)
{
    RandVec randVec;
    randVec.size = settings.crossoversPerGenerations * 2 * randVec.frequency;
    randVec.mul = settings.crossoversPerGenerations * 2; 

    int maxPopulation = settings.population + settings.crossoversPerGenerations * 2;
    int n = matrix.size;

    ScoreGenome* scoreGenomes = new ScoreGenome[maxPopulation];

    randVec.vec = new int[randVec.size];

    CudaMatrix cudaMatrix = ConvertToCudaMatrix(matrix);
    
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

    float best = scoreGenomes[0].score;

    for (int generation = 0; generation < settings.iterations; generation++)
    {
        if (generation % randVec.frequency == 0)
        {
            GenerateRandomVec(randVec.vec, randVec.size, 0, n);
        }

        int threadsPerBlock = 128;
        int blocks = (settings.crossoversPerGenerations + threadsPerBlock - 1) / threadsPerBlock;

        //cout << "Gen: " << generation << " | Best score: " << scoreGenomes[0].score << endl;

        ScoreGenome* cudaScoreGenomes;
        cudaMalloc(&cudaScoreGenomes, maxPopulation * sizeof(ScoreGenome));
        for (int i = 0; i < maxPopulation; i++)
        {
            int* d_g;
            cudaMalloc(&d_g, n * sizeof(int));
            cudaMemcpy(d_g, cudaScoreGenomes[i].genome.g, n * sizeof(int), cudaMemcpyHostToDevice);
        }

        CrossoverKernel << <blocks, threadsPerBlock >> > (
            scoreGenomes, &cudaMatrix, &randVec, &settings, generation);

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

    delete[] randVec.vec;

    FreeCudaMatrix(cudaMatrix);

    return result;

}


CudaMatrix* AllocateAndCopyCudaMatrixFromCPU(CudaMatrix& cudaMatrix)
{
    float* d_m;
    cudaMalloc(&d_m, cudaMatrix.size * sizeof(float));
    cudaMemcpy(d_m, cudaMatrix.m, cudaMatrix.size * sizeof(float), cudaMemcpyHostToDevice);

    CudaMatrix tmp;
    tmp.size = cudaMatrix.size;
    tmp.m = d_m;

    CudaMatrix* d_matrix;
    cudaMalloc(&d_matrix, sizeof(CudaMatrix));
    cudaMemcpy(d_matrix, &tmp, sizeof(CudaMatrix), cudaMemcpyHostToDevice);

    return d_matrix;
}

ScoreGenome* AllocateAndCopyScoreGenomesFromCPU(ScoreGenome* scoreGenomes, int size)
{
    ScoreGenome* d_scoreGenomes;
    cudaMalloc(&d_scoreGenomes, size * sizeof(ScoreGenome));

    for (int i = 0; i < size; i++)
    {
        int* d_g;
        cudaMalloc(&d_g, scoreGenomes[i].genome.size * sizeof(int));
        cudaMemcpy(d_g, scoreGenomes[i].genome.g, scoreGenomes[i].genome.size * sizeof(int), cudaMemcpyHostToDevice);

        ScoreGenome tmp = scoreGenomes[i];  
        tmp.genome.g = d_g;                 

        cudaMemcpy(&d_scoreGenomes[i], &tmp, sizeof(ScoreGenome), cudaMemcpyHostToDevice);
    }

    return d_scoreGenomes;
}

void CopyAndFreeScoreGenomesFromGPU(ScoreGenome* d_scoreGenomes, int size, ScoreGenome* scoreGenomes)
{
    for (int i = 0; i < size; i++)
    {
        ScoreGenome tmp;
        cudaMemcpy(&tmp, &d_scoreGenomes[i], sizeof(ScoreGenome), cudaMemcpyDeviceToHost);

        scoreGenomes[i].score = tmp.score;
        scoreGenomes[i].genome.size = tmp.genome.size;
        scoreGenomes[i].genome.g = new int[tmp.genome.size];

        cudaMemcpy(scoreGenomes[i].genome.g, tmp.genome.g, tmp.genome.size * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(tmp.genome.g);
    }

    cudaFree(d_scoreGenomes);

}

ScoreGenome CudaGenetic(Matrix& matrix, Settings settings)
{
    int threadsPerBlock = 128;
    int blocks = (settings.crossoversPerGenerations + threadsPerBlock - 1) / threadsPerBlock;

    int maxPopulation = settings.population + settings.crossoversPerGenerations * 2;
    int n = matrix.size;

    RandVec randVec;
    randVec.size = 10; //settings.crossoversPerGenerations * 2 * randVec.frequency;
    randVec.mul = settings.crossoversPerGenerations * 2;
    randVec.vec = new int[randVec.size];

    ScoreGenome* scoreGenomes = new ScoreGenome[maxPopulation];
    CudaMatrix cudaMatrix = ConvertToCudaMatrix(matrix);

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

    // --- GPU segment ---
    CudaMatrix* d_cudaMatrix = AllocateAndCopyCudaMatrixFromCPU(cudaMatrix);

    ScoreGenome* d_scoreGenomes = AllocateAndCopyScoreGenomesFromCPU(scoreGenomes, maxPopulation);

    int* d_randVec;
    cudaMalloc(&d_randVec, randVec.size);
    
    for (int generation = 0; generation < settings.iterations; generation++)
    {
        if (generation % randVec.frequency == 0)
                
        int* d_randVec; 
        cudaMalloc(&d_randVec, randVec.size * sizeof(int));
        GenerateRandomVec(d_randVec, randVec.size, 0, n);

        cudaMemcpy(randVec.vec, d_randVec, randVec.size * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < randVec.size; i++)
        {
            cout << randVec.vec[i] << endl;
        }

        cin.get();


    }



    CopyAndFreeScoreGenomesFromGPU(d_scoreGenomes, maxPopulation, scoreGenomes);

    // --- Back to CPU segment ---

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

    delete[] randVec.vec;

    delete[] cudaMatrix.m;

    return result;
}

