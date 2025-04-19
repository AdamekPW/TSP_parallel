#include "Cuda.cuh"



#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "\
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

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

void GenerateRandomVec(int* d_vec, int size, int lowerLimit, int upperLimit)
{
    RandomIntsKernel << <(size + 127) / 128, 128 >> > (d_vec, size, lowerLimit, upperLimit, time(NULL));
}

__device__ int GetRandomInt(int lower, int upper, unsigned long seed, int threadId = 0) {
    curandState state;
    curand_init(seed, threadId, 0, &state);

    float rand_uniform = curand_uniform(&state);
    int range = upper - lower;
    int randInt = lower + static_cast<int>(rand_uniform * range);

    if (randInt >= upper) randInt = upper - 1;

    return randInt;
}

__global__ void RandomIntsKernel(int* output, int n, int lower, int upper, unsigned long seed) {
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

__global__ void CrossoverGenomeCellKernel(int* randParentsVec, int* randCuttingPointsVec, int generation)
{
    int c = blockIdx.x;
    int i = threadIdx.x;
    
    generation = generation % d_params.randVecFrequency;

    int cuttingPoint = randCuttingPointsVec[generation * d_params.crossoversPerGeneration + c];

    int parent1 = randParentsVec[generation * d_params.randVecMul + 2 * c];
    int parent2 = randParentsVec[generation * d_params.randVecMul + 2 * c + 1];
    int child1 = d_params.population + 2 * c;
    int child2 = d_params.population + 2 * c + 1;

    __shared__ bool visited1[CROSSOVER_THREADS_PER_BLOCK];
    __shared__ bool visited2[CROSSOVER_THREADS_PER_BLOCK];
    
    int city1 = d_scoreGenomes[parent1].genome.g[i];
    int city2 = d_scoreGenomes[parent2].genome.g[i];
    
    if (i < cuttingPoint)
    {
        d_scoreGenomes[child1].genome.g[i] = city1;
        d_scoreGenomes[child2].genome.g[i] = city2;
        visited1[city1] = true;
        visited2[city2] = true;
    }
    else {
        visited1[city1] = false;
        visited2[city2] = false;
    }
    __syncthreads();
    
    if (i == 0)
    {
        int index = cuttingPoint;      
        for (int cityIndex = 0; cityIndex < CROSSOVER_THREADS_PER_BLOCK; cityIndex++)
        {
            int city = d_scoreGenomes[parent1].genome.g[cityIndex];
            if (!visited1[city])
            {
                d_scoreGenomes[child1].genome.g[index] = city;
                index++;
            }
        }
        d_scoreGenomes[child1].score = CudaScore(&d_scoreGenomes[child1].genome);
    }

    if (i == 1)
    {
        int index = cuttingPoint;
        for (int cityIndex = 0; cityIndex < CROSSOVER_THREADS_PER_BLOCK; cityIndex++)
        {
            int city = d_scoreGenomes[parent2].genome.g[cityIndex];
            if (!visited2[city])
            {
                d_scoreGenomes[child2].genome.g[index] = city;
                index++;
            }
        }
        d_scoreGenomes[child2].score = CudaScore(&d_scoreGenomes[child2].genome);
    }
      
}

__global__ void MutationKernel(int* d_mutationRandVec, int* d_randVec, int generation)
{
    //int m = blockIdx.x;
    int i = threadIdx.x;

    generation = generation % d_params.randVecFrequency;

    int prop = d_mutationRandVec[generation + i];

    int mutationArea = d_params.mutationProp * 10000;

    if (prop < 10000 - mutationArea) return; 

    if (prop >= 10000 - mutationArea + mutationArea / 2)
    {
        // tylko pierwszy w¹tek przeprowadza mutacje

        int index1 = GetRandomInt(0, d_params.n, 12345, i);
        int index2 = GetRandomInt(0, d_params.n, 12345, i);
        while (index1 == index2)
            index2 = GetRandomInt(0, d_params.n, 12345, i);

        int temp = d_scoreGenomes[i].genome.g[index1];
        d_scoreGenomes[i].genome.g[index1] = d_scoreGenomes[i].genome.g[index2];
        d_scoreGenomes[i].genome.g[index2] = temp;
    }
    else {

    }

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

__device__ float CudaScore(Genome* d_genome)
{
    float sum = 0;
    for (int i = 0; i < d_genome->size; i++)
    {
        int from = d_genome->g[i];
        int to = d_genome->g[(i + 1) % d_genome->size];
        sum += d_cudaMatrix.m[from * d_genome->size + to];
    }
    return sum;
}


void AllocateAndCopyCudaMatrix(CudaMatrix& cudaMatrix)
{
    float* d_m;
    cudaMalloc(&d_m, cudaMatrix.size * sizeof(float));

    // Skopiuj dane z hosta do GPU
    cudaMemcpy(d_m, cudaMatrix.m, cudaMatrix.size * sizeof(float), cudaMemcpyHostToDevice);

    // Utwórz tymczasow¹ strukturê na host
    CudaMatrix temp;
    temp.m = d_m;
    temp.size = cudaMatrix.size;

    // Skopiuj strukturê do zmiennej globalnej na GPU
    cudaMemcpyToSymbol(d_cudaMatrix, &temp, sizeof(CudaMatrix));
}

void AllocateAndCopyScoreGenomes(ScoreGenome* scoreGenomes, int size)
{
    // Tymczasowa CPU tablica do kopiowania struktur
    ScoreGenome* h_tmp = new ScoreGenome[size];

    for (int i = 0; i < size; ++i)
    {
        // Alokuj pamiêæ na genome.g na GPU
        int* d_g;
        cudaMalloc(&d_g, scoreGenomes[i].genome.size * sizeof(int));
        cudaMemcpy(d_g, scoreGenomes[i].genome.g, scoreGenomes[i].genome.size * sizeof(int), cudaMemcpyHostToDevice);

        // Skopiuj dane do tymczasowej struktury
        h_tmp[i].score = scoreGenomes[i].score;
        h_tmp[i].genome.size = scoreGenomes[i].genome.size;
        h_tmp[i].genome.g = d_g;
    }

    // Alokuj tablicê ScoreGenome na GPU
    ScoreGenome* d_tmp;
    cudaMalloc(&d_tmp, size * sizeof(ScoreGenome));
    cudaMemcpy(d_tmp, h_tmp, size * sizeof(ScoreGenome), cudaMemcpyHostToDevice);

    // Przypisz d_tmp do symbolu d_scoreGenomes
    cudaMemcpyToSymbol(d_scoreGenomes, &d_tmp, sizeof(ScoreGenome*));

    delete[] h_tmp;
}

void AllocateAndCopyParams(Params& params)
{
    cudaMemcpyToSymbol(d_params, &params, sizeof(Params));
}

void FreeAndCopyScoreGenomes(ScoreGenome* scoreGenomes, int size)
{
    // Pobierz wskaŸnik z symbolu
    ScoreGenome* d_ptr;
    cudaMemcpyFromSymbol(&d_ptr, d_scoreGenomes, sizeof(ScoreGenome*));

    for (int i = 0; i < size; ++i)
    {
        ScoreGenome tmp;
        cudaMemcpy(&tmp, &d_ptr[i], sizeof(ScoreGenome), cudaMemcpyDeviceToHost);

        scoreGenomes[i].score = tmp.score;
        scoreGenomes[i].genome.size = tmp.genome.size;

        // Skopiuj dane z GPU
        cudaMemcpy(scoreGenomes[i].genome.g, tmp.genome.g, tmp.genome.size * sizeof(int), cudaMemcpyDeviceToHost);

        // Zwolnij pamiêæ GPU alokowan¹ wczeœniej dla genome.g
        cudaFree(tmp.genome.g);
    }

    // Zwolnij tablicê struktur ScoreGenome na GPU
    cudaFree(d_ptr);
}


__device__ void bitonic_sort(ScoreGenome* data, int size, ScoreCompare comp) {
    unsigned int tid = threadIdx.x;

    for (unsigned int k = 2; k <= size; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = tid ^ j;

            if (ixj > tid && ixj < size) {
                bool ascending = ((tid & k) == 0);
                bool swap_needed = false;

                if (ascending) {
                    swap_needed = comp(data[ixj], data[tid]);
                }
                else {
                    swap_needed = comp(data[tid], data[ixj]);
                }

                if (swap_needed) {
                    ScoreGenome tmp = data[tid];
                    data[tid] = data[ixj];
                    data[ixj] = tmp;
                }
            }

            __syncthreads();
        }
    }
}

__global__ void sort_score_genomes(int size) {
    __shared__ ScoreGenome s_data[160];  // maksymalny wspierany rozmiar

    int tid = threadIdx.x;

    if (tid < size) {
        s_data[tid] = d_scoreGenomes[tid];
    }
    __syncthreads();

    ScoreCompare comp;
    bitonic_sort(s_data, size, comp);
    __syncthreads();

    if (tid < size) {
        d_scoreGenomes[tid] = s_data[tid];
    }
}



ScoreGenome CudaGenetic(Matrix& matrix, Settings settings)
{
    Params params;
    params.n = matrix.size;
    params.crossoversPerGeneration = settings.crossoversPerGenerations;
    params.population = settings.population;
    params.maxPopulation = settings.population + settings.crossoversPerGenerations * 2;
    params.mutationProp = settings.mutationProp;
    params.randVecFrequency = 1500;
    params.randVecMul = settings.crossoversPerGenerations * 2;
    params.randVecSize = params.randVecMul * params.randVecFrequency;

    ScoreGenome* scoreGenomes = new ScoreGenome[params.maxPopulation];
    CudaMatrix cudaMatrix = ConvertToCudaMatrix(matrix);

    for (int i = 0; i < params.maxPopulation; i++)
    {
        scoreGenomes[i].genome.g = new int[params.n];
        scoreGenomes[i].genome.size = params.n;
        SimpleSample(matrix, scoreGenomes[i].genome);
        scoreGenomes[i].score = Score(matrix, scoreGenomes[i].genome);
    }


    std::sort(scoreGenomes, scoreGenomes + params.maxPopulation, [](const ScoreGenome& a, const ScoreGenome& b) {
        return a.score < b.score;
    });
    cout << "Start best: " << scoreGenomes[0].score << endl;

    // --- GPU segment ---

    AllocateAndCopyCudaMatrix(cudaMatrix);

    AllocateAndCopyScoreGenomes(scoreGenomes, params.maxPopulation);
    
    AllocateAndCopyParams(params);

    int* d_randVec;
    cudaMalloc(&d_randVec, params.randVecSize * sizeof(int));

    int* d_randCuttingPointVec;
    cudaMalloc(&d_randCuttingPointVec, (int)(params.randVecSize / 2) * sizeof(int));
    
    int* d_mutationRandVec;
    cudaMalloc(&d_mutationRandVec, params.maxPopulation * params.randVecFrequency * sizeof(int));

    for (int generation = 0; generation < settings.iterations; generation++)
    {
        //cout << generation << endl;
        //if (generation % params.randVecFrequency == 0)
        //{
        //    GenerateRandomVec(d_randVec, params.randVecSize, 0, params.n);
        //    GenerateRandomVec(d_randCuttingPointVec, (int)(params.randVecSize / 2), 1, params.n - 1);
        //    GenerateRandomVec(d_mutationRandVec, params.maxPopulation * params.randVecFrequency, 0, 10000 - params.mutationProp * 10000);
        //}
        //
        //CrossoverGenomeCellKernel << < params.crossoversPerGeneration, params.n >> >
        //    (d_randVec, d_randCuttingPointVec, generation);

        ////MutationKernel(int* d_mutationRandVec, int* d_randVec, int generation)
        ////MutationKernel << < 1, params.maxPopulation>> > (d_mutationRandVec, d_randVec, generation);

        //sort_score_genomes << <1, 256 >> > (params.maxPopulation);
    
        float currentBest;
        cudaMemcpy(&currentBest, &(d_scoreGenomes[0].score), sizeof(float), cudaMemcpyDeviceToHost);
        cout << currentBest << endl;

    }


    cudaFree(&d_randVec);

    cudaFree(&d_randCuttingPointVec);

    cudaFree(&d_mutationRandVec);

    for (int i = 0; i < params.maxPopulation; i++)
    {
        scoreGenomes[i].score = 0;
        scoreGenomes[i].genome.size = 0;
        for (int j = 0; j < params.n; j++)
        {
            scoreGenomes[i].genome.g[j] = 0;         
        }
    }

    FreeAndCopyScoreGenomes(scoreGenomes, params.maxPopulation);


    // --- Back to CPU segment ---

    ScoreGenome result;
    result.score = scoreGenomes[0].score;
    result.genome.size = scoreGenomes->genome.size;
    result.genome.g = new int[scoreGenomes->genome.size];

    for (int i = 0; i < params.n; i++)
    {
        result.genome.g[i] = scoreGenomes[0].genome.g[i];
    }

    for (int i = 0; i < params.maxPopulation; i++)
    {
        delete[] scoreGenomes[i].genome.g;
    }

    delete[] scoreGenomes;

    delete[] cudaMatrix.m;

    return result;
}

