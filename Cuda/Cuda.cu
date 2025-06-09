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



__device__ CudaMatrix d_cudaMatrix;
CudaMatrix* d_cudaMatrix_ptr = nullptr;



void AllocateCudaMatrix(int size) {
    CudaMatrix h_temp;
    h_temp.size = size;

    size_t totalSize = size * size * sizeof(float);  // Zak³adam macierz kwadratow¹
    CUDA_CHECK(cudaMalloc(&h_temp.m, totalSize));

    // Skopiuj strukturê (z wskaŸnikiem do pamiêci urz¹dzenia) do zmiennej globalnej w GPU
    CUDA_CHECK(cudaMemcpyToSymbol(d_cudaMatrix, &h_temp, sizeof(CudaMatrix)));
}

void FreeCudaMatrix() {
    CudaMatrix h_temp;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_temp, d_cudaMatrix, sizeof(CudaMatrix)));

    CUDA_CHECK(cudaFree(h_temp.m));
}

void CopyCudaMatrixFromHostToDevice(Matrix& matrix) {
    int n = matrix.size;
    float* host_flat = new float[n * n];

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            host_flat[i * n + j] = matrix.m[i][j];
        }
    }

    CudaMatrix h_temp;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_temp, d_cudaMatrix, sizeof(CudaMatrix)));

    CUDA_CHECK(cudaMemcpy(h_temp.m, host_flat, n * n * sizeof(float), cudaMemcpyHostToDevice));

    delete[] host_flat;
}

__device__ int DeviceRandomNumber(int lowerLimit, int upperLimit, int idx, unsigned int seed)
{
    curandState state;
    curand_init(seed, idx, 0, &state);
    int range = upperLimit - lowerLimit;
    if (range <= 0) return lowerLimit;  // zabezpieczenie
    int randVal = curand(&state) % range + lowerLimit;
    return randVal;
}

__device__ int DeviceRandomNumber(int lowerLimit, int upperLimit, curandState* state)
{
    int range = upperLimit - lowerLimit;
    if (range <= 0) return lowerLimit;  
    int randVal = curand(state) % range + lowerLimit;
    return randVal;
}

__global__ void InitPopulationKernel(int* d_population, int maxPopulation, int n, curandState* statesTable)
{
    int sequenceId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sequenceId >= maxPopulation) return;

    int Start = DeviceRandomNumber(0, n, &statesTable[sequenceId * n]);
    bool Visited[N] = { false };
    Visited[Start] = true;
    int VisitedCount = 1;

    int i = Start;
    while (VisitedCount <= n)
    {
        d_population[sequenceId * n + VisitedCount - 1] = i;
        Visited[i] = true;

        float min1Value = (float)INT_MAX;
        int min1Index = -1;
        float min2Value = (float)INT_MAX;
        int min2Index = -1;

        for (int j = 0; j < n; j++)
        {

            float cMatrixValue = d_cudaMatrix.m[i * n + j];
            if (cMatrixValue > 0 && !Visited[j] && cMatrixValue <= min1Value)
            {
                min2Value = min1Value;
                min2Index = min1Index;
                min1Value = cMatrixValue;
                min1Index = j;
            }
        }
        int Randomize = DeviceRandomNumber(0, 3, &statesTable[sequenceId * n]);
        if (Randomize == 0 && min2Index != -1)
        {
            i = min2Index;
        }
        else
        {
            i = min1Index;

        }
        VisitedCount++;
    }
}

__device__ float GetScore(int* sequence, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        int from = sequence[i];
        int to = sequence[(i + 1) % n];
        sum += d_cudaMatrix.m[from * n + to];
    }
    return sum;
}

__device__ bool IsConsistent(int* sequence, int n)
{
    bool IsInSequence[N] = { false };

    for (int i = 0; i < n; i++)
    {
        if (IsInSequence[sequence[i]]) return false;

        IsInSequence[sequence[i]] = true;
    }
    return true;
}

__global__ void InitRandomStatesKernel(curandState* states,  unsigned int seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, id, 0, &states[id]);
}

__device__ __host__ inline int posmod(int a, int b) 
{
    return (a % b + b) % b;
}

__device__ __host__ inline int ABS(int a) 
{
    return a < 0 ? -a : a;
}

__device__ __host__ inline int MinDistance(int a, int b, int n) 
{
    int d = abs(a - b);
    return d < (n - d) ? d : (n - d);
}

__device__ __host__ inline void Swap(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void Crossover(int* g1_in, int* g2_in, int* g1_out, int* g2_out, int n)
{
    int cuttingPoint = RandomNumber(1, n - 1);

    bool visited1[MAX_N] = { false };
    bool visited2[MAX_N] = { false };

    // Kopiuj lewe czesci od cuttingPoint do potomstwa
    for (int i = 0; i < cuttingPoint; i++)
    {
        int city1 = g1_in[i];
        int city2 = g2_in[i];
        g1_out[i] = city1;
        g2_out[i] = city2;
        visited1[city1] = true;
        visited2[city2] = true;
    }

    int g1_out_index = cuttingPoint;
    int g2_out_index = cuttingPoint;


    // Kopiuj kolejno pozostale, o ile nie wystepuja w potomstwie
    for (int i = 0; i < n; i++)
    {
        int city1 = g2_in[i];
        int city2 = g1_in[i];
        if (!visited1[city1])
        {
            g1_out[g1_out_index] = city1;
            g1_out_index++;
            visited1[city1] = true;
        }
        if (!visited2[city2])
        {
            g2_out[g2_out_index] = city2;
            g2_out_index++;
            visited2[city2] = true;
        }
    }

}

__global__ void CrossoverKernel(int* population, int crossoverCount, int populationCount, int n, curandState* statesTable)
{
    int c = blockIdx.x; // ka¿dy blok odpowiada za jedno krzy¿owanie

    int i = threadIdx.x; // ka¿dy w¹tek odpowiada za jedno miasto

    __shared__ int cuttingPoint;
    __shared__ int parent1, parent2, child1, child2;

    if (i == 0)
    {
        curandState localState = statesTable[c * n];
        parent1 = DeviceRandomNumber(0, populationCount, &localState);
        parent2 = DeviceRandomNumber(0, populationCount, &localState);
        while (parent1 == parent2) parent2 = DeviceRandomNumber(0, populationCount, &localState);
        cuttingPoint = DeviceRandomNumber(1, n - 1, &localState);

        statesTable[c * n] = localState;

        child1 = populationCount + 2 * c;
        child2 = populationCount + 2 * c + 1;
    }

    __syncthreads();

    __shared__ bool visited1[N];
    __shared__ bool visited2[N];

    int city1 = population[parent1 * n + i];
    int city2 = population[parent2 * n + i];

    if (i < cuttingPoint)
    {
        population[child1 * n + i] = city1;
        population[child2 * n + i] = city2;
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
        for (int cityIndex = 0; cityIndex < n; cityIndex++)
        {
            int city = population[parent1 * n + cityIndex];
            if (!visited1[city])
            {
                population[child1 * n + index] = city;
                index++;
            }
        }
        //printf("%f\n", d_scoreGenomes[child1].score);
    }

    if (i == 1)
    {
        int index = cuttingPoint;
        for (int cityIndex = 0; cityIndex < n; cityIndex++)
        {
            int city = population[parent2 * n + cityIndex];
            if (!visited2[city])
            {
                population[child2 * n + index] = city;
                index++;
            }
        }

    }
    __syncthreads();

}

__global__ void MutationKernel(int* population, int n, curandState* d_curandStates)
{
    int sequenceId = blockIdx.x;
    int i = threadIdx.x;

    __shared__ float Results[1024];
    __shared__ int SequenceLocal[N];

    if (i < n)
    {
        SequenceLocal[i] = population[sequenceId * n + i];
    }

    __syncthreads();

    int index1 = DeviceRandomNumber(0, n, &d_curandStates[sequenceId * n]);
    int index2 = DeviceRandomNumber(0, n, &d_curandStates[sequenceId * n]);
    
    if (index1 == index2) index1 = (index2 + 1) % n;

    if (MinDistance(index1, index2, n) == 1) index2 = (index2 + 3) % n;

    int left_city_1 = SequenceLocal[posmod(index1 - 1, n)];
    int mid_city_1 = SequenceLocal[posmod(index1, n)];  
    int right_city_1 = SequenceLocal[posmod(index1 + 1, n)];
    int left_city_2 = SequenceLocal[posmod(index2 - 1, n)];
    int mid_city_2 = SequenceLocal[posmod(index2, n)];
    int right_city_2 = SequenceLocal[posmod(index2 + 1, n)];
    
    float originalPathValue = 0.0f;
    float mutationPathValue = 0.0f;

    originalPathValue = d_cudaMatrix.m[left_city_1 * n + mid_city_1] + d_cudaMatrix.m[mid_city_1 * n + right_city_1]
        + d_cudaMatrix.m[left_city_2 * n + mid_city_2] + d_cudaMatrix.m[mid_city_2 * n + right_city_2];
    mutationPathValue = d_cudaMatrix.m[left_city_1 * n + mid_city_2] + d_cudaMatrix.m[mid_city_2 * n + right_city_1]
        + d_cudaMatrix.m[left_city_2 * n + mid_city_1] + d_cudaMatrix.m[mid_city_1 * n + right_city_2];
 
    Results[i] = originalPathValue - mutationPathValue;
    

    __syncthreads();

    int currMaxIndex = 0; 
    for (int j = 0; j < n; j++)
    {
        if (Results[j] > Results[currMaxIndex])
        {
            currMaxIndex = j;
        }
    }

    if (i == currMaxIndex && Results[currMaxIndex] > 0)
    {
        int id1 = sequenceId * n + index1;
        int id2 = sequenceId * n + index2;
        int temp = population[id1];
        population[id1] = population[id2];
        population[id2] = temp;
    }

    __syncthreads();
}

__global__ void GetScoresKernel(int* population, int n, float* scores, int scoresLength)
{
    int sequenceId = blockDim.x * blockIdx.x + threadIdx.x;

    if (sequenceId >= scoresLength) return;

    if (!IsConsistent(&population[sequenceId * n], n))
    {
        printf("Wykryto niespojnosc w sekwencji: %d\n", sequenceId);
    }
    scores[sequenceId] = GetScore(&population[sequenceId * n], n);
}

void sort_by_cost(int* h_population, float* h_costs, int n, int population_size) {
    std::vector<int> indices(population_size);
    for (int i = 0; i < population_size; ++i)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return h_costs[a] < h_costs[b]; 
        });

    std::vector<int> sorted_population(n * population_size);
    std::vector<float> sorted_costs(population_size);

    for (int i = 0; i < population_size; ++i) {
        int src_index = indices[i];
        std::memcpy(&sorted_population[i * n], &h_population[src_index * n], n * sizeof(int));
        sorted_costs[i] = h_costs[src_index];
    }

    std::memcpy(h_population, sorted_population.data(), n * population_size * sizeof(int));
    std::memcpy(h_costs, sorted_costs.data(), population_size * sizeof(float));
}

ScoreGenome CudaGenetic(Matrix &matrix, Settings settings)
{
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    //printf("Max threads per block: %d\n", maxThreadsPerBlock);

    unsigned int seed = static_cast<unsigned int>(time(NULL));


    int iterations = settings.iterations;
    int crossoversPerGeneration = settings.crossoversPerGenerations;
    int population = settings.population;
    int maxPopulation = settings.population + settings.crossoversPerGenerations * 2;

    int n = matrix.size;
    int d_stateTableSize = maxPopulation * n; // ka¿dy w¹tek w generacji otrzyma w³asny curandState

    int* h_population = new int[n*maxPopulation];

    int* d_population;

    int totalPopulationSize = n * maxPopulation * sizeof(int);

    cudaMalloc(&d_population, totalPopulationSize);

    float* h_scores = new float[maxPopulation];

    float* d_scores;

    cudaMalloc(&d_scores, maxPopulation * sizeof(float));

    curandState* d_stateTable;
    
    CUDA_CHECK(cudaMalloc(&d_stateTable, d_stateTableSize * sizeof(curandState)));

    AllocateCudaMatrix(n * n);
    CopyCudaMatrixFromHostToDevice(matrix);

    InitRandomStatesKernel << <(d_stateTableSize + maxThreadsPerBlock - 1) / maxThreadsPerBlock, maxThreadsPerBlock >> > (d_stateTable, seed);

    InitPopulationKernel << < (maxPopulation + maxThreadsPerBlock - 1) / maxThreadsPerBlock, maxThreadsPerBlock >> > (d_population, maxPopulation, n, d_stateTable);

    GetScoresKernel << < maxPopulation, 1 >> > (d_population, n, d_scores, maxPopulation);

    cudaMemcpy(h_scores, d_scores, maxPopulation * sizeof(float), cudaMemcpyDeviceToHost);

    float AVG = 0.0f;
    float Best = (float)INT_MAX;
    for (int i = 0; i < maxPopulation; i++)
    {
        AVG += h_scores[i];
        if (h_scores[i] < Best) 
        {
            Best = h_scores[i];
        }
    }

    //cout << "Najlepszy wynik przed: " << Best << " | Sredni wynik przed: " << AVG / (float)params.maxPopulation << endl;

    int crossoverFreq = 10;

    for (int generation = 0; generation < settings.iterations ; generation++)
    {
        if (generation % crossoverFreq == 0)
        {
            GetScoresKernel << < maxPopulation, 1 >> > (d_population, n, d_scores, maxPopulation);
    
            cudaMemcpy(h_scores, d_scores, maxPopulation, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_population, d_population, totalPopulationSize, cudaMemcpyDeviceToHost);
    
            // sortowanie
            sort_by_cost(h_population, h_scores, n, maxPopulation);

            // generacje
            #pragma omp parallel for default(none) shared(population, h_population, crossoversPerGeneration, n) 
            for (int i = 0; i < crossoversPerGeneration; i++)
            {
                int pIndex1 = RandomNumber(0, population);
                int pIndex2 = RandomNumber(0, population);
                while (pIndex1 == pIndex2) pIndex2 = RandomNumber(0, population);

                int cIndex1 = population + 2 * i;
                int cIndex2 = cIndex1 + 1;

                Crossover(&h_population[pIndex1 * n], &h_population[pIndex2 * n], &h_population[cIndex1 * n], &h_population[cIndex2 * n], n);
            }

            cudaMemcpy(d_population, h_population, totalPopulationSize, cudaMemcpyHostToDevice);
        }
        // mutacje
        MutationKernel << < maxPopulation, MAX_THREADS>> > (d_population, n, d_stateTable);
    }

    GetScoresKernel << < maxPopulation, 1 >> > (d_population, n, d_scores, maxPopulation);

    cudaMemcpy(h_scores, d_scores, maxPopulation * sizeof(float), cudaMemcpyDeviceToHost);

    AVG = 0.0f;
    Best = (float)INT_MAX;
    int BestIndex;
    ScoreGenome result;
    for (int i = 0; i < maxPopulation; i++)
    {
        AVG += h_scores[i];
        if (h_scores[i] < Best)
        {
            Best = h_scores[i];
            BestIndex = i;
        }
    }

    result.score = Best;
    result.genome.size = n;
    result.genome.g = new int[n];
    cudaMemcpy(result.genome.g, &d_population[BestIndex * n], n * sizeof(int), cudaMemcpyDeviceToHost);

    //cout << "Najlepszy wynik po: " << Best << " | Sredni wynik po: " << AVG / (float)params.maxPopulation << endl;


    cudaDeviceSynchronize();

    FreeCudaMatrix();

    cudaFree(d_population);

    cudaFree(d_scores);

    CUDA_CHECK(cudaFree(d_stateTable));

    delete[] h_scores;

    delete[] h_population;

    return result;
}